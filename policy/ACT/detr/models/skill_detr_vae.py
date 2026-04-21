# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Skill-augmented ACT model.
"""
import torch
from torch import nn

from .backbone import build_backbone, build_tactile_backbone
from .transformer import build_transformer
from .detr_vae import reparametrize, get_sinusoid_encoding_table, build_encoder, mlp


class SkillDETRVAE(nn.Module):
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, tactile_names, num_skills):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.transformer = transformer
        self.encoder = encoder
        self.num_skills = num_skills

        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.vision_input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.tactile_input_proj = nn.Conv2d(backbones[1].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            raise NotImplementedError('SkillDETRVAE currently expects visual/tactile backbones.')

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # latent + proprio

        self.skill_context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.skill_head = mlp(hidden_dim, hidden_dim, num_skills, hidden_depth=2)
        self.skill_token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.skill_pos_embed = nn.Embedding(1, hidden_dim)

    def _encode_latent(self, qpos, actions=None, is_pad=None):
        bs, _ = qpos.shape
        if actions is not None:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1).permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            enc_is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=enc_is_pad)[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
        latent_input = self.latent_out_proj(latent_sample)
        return latent_input, mu, logvar

    def _encode_observations(self, qpos, cam_image, tac_image):
        all_features = []
        all_pos = []

        for cam_id, _ in enumerate(self.camera_names):
            features, pos = self.backbones[0](cam_image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_features.append(self.vision_input_proj(features).flatten(2))
            all_pos.append(pos.flatten(2))

        for tac_id, _ in enumerate(self.tactile_names):
            features, pos = self.backbones[1](tac_image[:, tac_id])
            features = features[0]
            pos = pos[0]
            all_features.append(self.tactile_input_proj(features).flatten(2))
            all_pos.append(pos.flatten(2))

        proprio_input = self.input_proj_robot_state(qpos)
        src = torch.cat(all_features, axis=2)
        pos = torch.cat(all_pos, axis=2)
        return src, pos, proprio_input

    def _build_skill_branch(self, src, proprio_input):
        pooled_obs = src.mean(dim=2)
        skill_context = self.skill_context_proj(torch.cat([pooled_obs, proprio_input], dim=1))
        skill_logits = self.skill_head(skill_context)
        skill_token = self.skill_token_proj(skill_context)
        return skill_logits, skill_token

    def _decode(self, src, pos, latent_input, proprio_input, skill_token):
        bs, _, _ = src.shape
        src_tokens = src.permute(2, 0, 1)
        if pos.shape[0] == 1:
            pos_tokens = pos.permute(2, 0, 1).repeat(1, bs, 1)
        else:
            pos_tokens = pos.permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        extra_tokens = torch.stack([skill_token, latent_input, proprio_input], axis=0)
        extra_pos = torch.cat([
            self.skill_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1),
            self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1),
        ], axis=0)

        encoder_src = torch.cat([extra_tokens, src_tokens], axis=0)
        encoder_pos = torch.cat([extra_pos, pos_tokens], axis=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.transformer.encoder(encoder_src, pos=encoder_pos)
        hs = self.transformer.decoder(tgt, memory, pos=encoder_pos, query_pos=query_embed)
        return hs.transpose(1, 2)

    def forward(self, qpos, cam_image, tac_image, env_state, actions=None, is_pad=None):
        latent_input, mu, logvar = self._encode_latent(qpos, actions, is_pad)
        src, pos, proprio_input = self._encode_observations(qpos, cam_image, tac_image)
        skill_logits, skill_token = self._build_skill_branch(src, proprio_input)
        hs = self._decode(src, pos, latent_input, proprio_input, skill_token)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], skill_logits


def build(args):
    state_dim = args.state_dim
    num_skills = getattr(args, 'num_skills', None)
    if num_skills is None:
        raise ValueError('SkillACT requires `num_skills` in the config.')

    backbones = [build_backbone(args), build_tactile_backbone(args)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = SkillDETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        tactile_names=args.tactile_names,
        num_skills=num_skills,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of parameters: %.2fM' % (n_parameters / 1e6, ))
    return model
