import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .base import TrainerX
from .optim import build_optimizer
from collections import OrderedDict
import json

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # Meta-Net setting
        vis_dim = clip_model.visual.output_dim 
        self.bias_meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim * 2, cfg.layer1)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(cfg.layer1, cfg.layer2)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(cfg.layer2, cfg.layer3)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear4", nn.Linear(cfg.layer3, ctx_dim))
        ]))
        self.bias_meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # Load VOCAB and generate prompts
        if cfg.DATASET.NAME == "ImageNetA" or cfg.DATASET.NAME == "ImageNetR" or cfg.DATASET.NAME == "ImageNetSketch" or cfg.DATASET.NAME == "ImageNetV2":
            with open(f"./VOCAB/{cfg.VOCAB}/ImageNet_{cfg.SEED}.json", 'r') as file:
                data = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/ImageNet_1.json", 'r') as file:
                data1 = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/ImageNet_2.json", 'r') as file:
                data2 = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/ImageNet_3.json", 'r') as file:
                data3 = json.load(file)
        else:
            with open(f"./VOCAB/{cfg.VOCAB}/{cfg.DATASET.NAME}_{cfg.SEED}.json", 'r') as file:
                data = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/{cfg.DATASET.NAME}_1.json", 'r') as file:
                data1 = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/{cfg.DATASET.NAME}_2.json", 'r') as file:
                data2 = json.load(file)
            with open(f"./VOCAB/{cfg.VOCAB}/{cfg.DATASET.NAME}_3.json", 'r') as file:
                data3 = json.load(file)
        prompts = [prompt_prefix + " " + name + " " + ' '.join(data[name].split()[:cfg.NUM_A]) + "." for name in classnames]
        prompts1 = [prompt_prefix + " " + name + " " + ' '.join(data1[name].split()[:cfg.NUM_A]) + "." for name in classnames]
        prompts2 = [prompt_prefix + " " + name + " " + ' '.join(data2[name].split()[:cfg.NUM_A]) + "." for name in classnames]
        prompts3 = [prompt_prefix + " " + name + " " + ' '.join(data3[name].split()[:cfg.NUM_A]) + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        tokenized_prompts1 = torch.cat([clip.tokenize(p) for p in prompts1])
        tokenized_prompts2 = torch.cat([clip.tokenize(p) for p in prompts2])
        tokenized_prompts3 = torch.cat([clip.tokenize(p) for p in prompts3])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            embedding1 = clip_model.token_embedding(tokenized_prompts1).type(dtype)
            embedding2 = clip_model.token_embedding(tokenized_prompts2).type(dtype)
            embedding3 = clip_model.token_embedding(tokenized_prompts3).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_suffix1", embedding1[:, 1 + n_ctx:, :])
        self.register_buffer("token_suffix2", embedding2[:, 1 + n_ctx:, :])
        self.register_buffer("token_suffix3", embedding3[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_prompts1 = tokenized_prompts1
        self.tokenized_prompts2 = tokenized_prompts2
        self.tokenized_prompts3 = tokenized_prompts3
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix1 = self.token_suffix1
        suffix2 = self.token_suffix2
        suffix3 = self.token_suffix3

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts1 = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix1,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts2 = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix2,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts3 = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix3,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, prompts1, prompts2, prompts3
    
    def bias_term(self, image_features, text_features):
        fusion_features = torch.cat((text_features, image_features.repeat(text_features.shape[0],1)), dim=1)
        bias = self.bias_meta_net(fusion_features)
        return text_features + bias


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts1 = self.prompt_learner.tokenized_prompts1
        self.tokenized_prompts2 = self.prompt_learner.tokenized_prompts2
        self.tokenized_prompts3 = self.prompt_learner.tokenized_prompts3
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts, prompts1, prompts2, prompts3 = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts1 = self.tokenized_prompts1
        tokenized_prompts2 = self.tokenized_prompts2
        tokenized_prompts3 = self.tokenized_prompts3

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits = []
        if self.prompt_learner.training:
            for imf_i in image_features:
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)    
                text_features = self.prompt_learner.bias_term(imf_i, text_features)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
        else:
            if self.subsample_classes == 'base':
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                for imf_i in image_features:
                    bias_text_features = self.prompt_learner.bias_term(imf_i, text_features)
                    l_i = logit_scale * imf_i @ bias_text_features.t()
                    logits.append(l_i)
                logits = torch.stack(logits)
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
                for imf_i in image_features:
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


@TRAINER_REGISTRY.register()
class CoOp_CoAPT(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # self.model.to(self.device)
        # # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.model.to(self.device)
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_suffix1" in state_dict:
                del state_dict["token_suffix1"]
            if "token_suffix2" in state_dict:
                del state_dict["token_suffix2"]
            if "token_suffix3" in state_dict:
                del state_dict["token_suffix3"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
