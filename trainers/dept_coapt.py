import copy
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler

from .optim import build_optimizer
from .dept import FiLM
from .promptsrc_coapt import CustomCLIP as CustomCLIP_, PromptSRC_CoAPT, load_clip_to_cpu


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.lp_cfg = cfg.TRAINER.LINEAR_PROBE
        self.film_cfg = cfg.TRAINER.FILM

        clip_dim = clip_model.text_projection.size(1)
        
        film_cfg = self.film_cfg

        if film_cfg.LINEAR_PROBE:
            self.film_lp_img = FiLM(clip_dim)
            self.film_lp_text = FiLM(clip_dim)
        
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            self.linear_probe_proj = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        else:
            self.linear_probe_proj = nn.Identity()
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        tokenized_prompts = self.tokenized_prompts
        prompts, _, _, _ = self.prompt_learner()

        # text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))

        # text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * img_feats_norm @ text_feats_norm.t()
        
        logits = []
        if self.prompt_learner.training:
            total_text_feats = []
            for imf_i in img_feats_norm:
                text_feats = self.text_encoder(prompts, tokenized_prompts)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                text_feats = self.prompt_learner.bias_term(imf_i, text_feats)
                total_text_feats.append(text_feats)
                l_i = logit_scale * imf_i @ text_feats.t()
                logits.append(l_i)
            total_text_feats = torch.stack(total_text_feats)
            total_text_feats = total_text_feats.mean(dim=0)

            logits = torch.stack(logits)
            zs_text_feats = self.prompt_learner.fixed_embeddings
            zs_text_feats_norm = zs_text_feats / zs_text_feats.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                zs_img_feats = self.prompt_learner.ZS_image_encoder(img.type(self.dtype))
                zs_img_feats_norm = zs_img_feats / zs_img_feats.norm(dim=-1, keepdim=True)
                zs_logits = logit_scale * zs_img_feats_norm.cuda() @ zs_text_feats_norm.half().cuda().t()

            logits_lp, labels_lp = self._forward_logits_linear_probe(total_text_feats, img_feats, labels, 
                                                                     zs_text_feats, zs_img_feats)

            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            zs_text_feats = zs_text_feats / zs_text_feats.norm(dim=-1, keepdim=True)
            zs_img_feats = zs_img_feats / zs_img_feats.norm(dim=-1, keepdim=True)

            return self._loss(logits, labels, logits_lp, labels_lp), \
                   total_text_feats, zs_text_feats, zs_img_feats, img_feats, zs_logits, logits
        else:
            text_feats = self.text_encoder(prompts, tokenized_prompts)
            text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
            total_text_feats = []
            for imf_i in img_feats_norm:
                bias_text_feats = self.prompt_learner.bias_term(imf_i, text_feats_norm)
                total_text_feats.append(bias_text_feats)
                l_i = logit_scale * imf_i @ bias_text_feats.t()
                logits.append(l_i)
            total_text_feats = torch.stack(total_text_feats)
            total_text_feats = total_text_feats.mean(dim=0)
            logits = torch.stack(logits)

            logits_lp, _ = self._forward_logits_linear_probe(total_text_feats, img_feats)

            if not self.lp_cfg.TEST_TIME_FUSION:
                return logits_lp

            lp_weight = self.lp_cfg.WEIGHT
            logits = (1 - lp_weight) * logits + lp_weight * logits_lp
            return logits
    
    def _forward_new(self, img):
        assert not self.prompt_learner.training
        logit_scale = self.logit_scale.exp()
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts1 = self.tokenized_prompts1
        tokenized_prompts2 = self.tokenized_prompts2
        tokenized_prompts3 = self.tokenized_prompts3
        prompts, prompts1, prompts2, prompts3 = self.prompt_learner()

        img_feats = self.image_encoder(img.type(self.dtype))
        img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)

        if self.subsample_classes == 'base':
            text_feats = self.text_encoder(prompts, tokenized_prompts)
            text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
            logits = logit_scale * img_feats_norm @ text_feats_norm.t()
        else:
            logits1 = []
            logits2 = []
            logits3 = []

            text_features1 = self.text_encoder(prompts1, tokenized_prompts1)
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            
            text_features2 = self.text_encoder(prompts2, tokenized_prompts2)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            
            text_features3 = self.text_encoder(prompts3, tokenized_prompts3)
            text_features3 = text_features3 / text_features3.norm(dim=-1, keepdim=True)
            for imf_i in img_feats_norm:
                bias_text_features1 = self.prompt_learner.bias_term(imf_i, text_features1)
                l1_i = logit_scale * imf_i @ bias_text_features1.t()
                logits1.append(l1_i)
                
                bias_text_features2 = self.prompt_learner.bias_term(imf_i, text_features2)
                l2_i = logit_scale * imf_i @ bias_text_features2.t()
                logits2.append(l2_i)
                
                bias_text_features3 = self.prompt_learner.bias_term(imf_i, text_features3)
                l3_i = logit_scale * imf_i @ bias_text_features3.t()
                logits3.append(l3_i)
            logits1 = torch.stack(logits1)
            logits2 = torch.stack(logits2)
            logits3 = torch.stack(logits3)
            logits = logits1 + logits2 + logits3

        return logits


    def _forward_logits_linear_probe(self, text_feats, img_feats, labels=None, 
                                     zs_text_feats=None, zs_img_feats=None):
        if self.film_cfg.LINEAR_PROBE:
            text_feats = self.film_lp_text(text_feats)
            img_feats = self.film_lp_img(img_feats)

        if labels is None:
            all_feats = img_feats
            all_labels = labels
        else:
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_logits = self.linear_probe_proj(all_feats)
        return all_logits, all_labels

    def _loss(self, logits, labels, logits_lp, labels_lp):
        loss_cls = F.cross_entropy(logits, labels)
        loss_cls_lp = F.cross_entropy(logits_lp, labels_lp)

        cls_weight = self.lp_cfg.CLS_WEIGHT
        lp_weight = self.lp_cfg.WEIGHT
        # loss = (1 - lp_weight) * loss_cls + lp_weight * loss_cls_lp
        loss = cls_weight * loss_cls + lp_weight * loss_cls_lp
        return loss


@TRAINER_REGISTRY.register()
class DePT_CoAPT(PromptSRC_CoAPT):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break

            if "ZS_image_encoder" in name:
                update = False
                
            param.requires_grad_(update)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {list(sorted(enabled))}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # Keep model with GPA
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)
        
        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)

        return loss_summary
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)
                
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            if "prompt_learner.token_suffix1" in state_dict:
                del state_dict["prompt_learner.token_suffix1"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]
            if "prompt_learner.token_suffix3" in state_dict:
                del state_dict["prompt_learner.token_suffix3"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
                from datasets.imagenet import ImageNet
                from dassl.utils import listdir_nohidden

                dataset = self.dm.dataset
                text_file = osp.join(dataset.dataset_dir, "classnames.txt")
                all_folders = ImageNet.read_classnames(text_file).keys()

                TO_BE_IGNORED = ["README.txt"]
                folders = listdir_nohidden(dataset.image_dir, sort=True)
                folders = [f for f in folders if f not in TO_BE_IGNORED]
                is_reserves = [f in folders for f in all_folders]

                print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
                state_dict['linear_probe_proj.weight'] = state_dict['linear_probe_proj.weight'][is_reserves]
                state_dict['linear_probe_proj.bias'] = state_dict['linear_probe_proj.bias'][is_reserves]
            
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)