import torch
import argparse
import itertools
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from models.modules.helpers_3detr import GenericMLP
from torch.cuda.amp import autocast
from models import Res16UNet34C
from models.modules.attention import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.position_embedding import PositionalEncoding1D, PositionEmbeddingCoordsSine, PositionalEncoding3D
from models.text_encoder import TextEncoder


class Interactive4D(nn.Module):
    def __init__(
        self,
        num_heads,
        num_decoders,
        hidden_dim,
        dim_feedforward,
        shared_decoder,
        num_bg_queries,
        dropout,
        pre_norm,
        aux,
        voxel_size,
        sample_sizes,
        sweep_size,
        text_encoder_backend="hash_ngram",
        text_encoder_model_name_or_path=None,
        freeze_text_encoder=False,
        text_encoder_vocab_size=8192,
        text_encoder_dim=256,
    ):
        super().__init__()

        backbone_cfg = argparse.ArgumentParser()
        backbone_cfg.dilations = [1, 1, 1, 1]
        backbone_cfg.conv1_kernel_size = 5
        backbone_cfg.bn_momentum = 0.02
        if sweep_size == 1:
            in_channels = 2
        else:
            in_channels = 3  # scan index is added as an additional feature
        self.is_cuda_available = torch.cuda.is_available()
        self.sweep_size = sweep_size
        self.backbone = Res16UNet34C(in_channels=in_channels, out_channels=19, config=backbone_cfg)
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        self.mask_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.shared_decoder = shared_decoder
        self.num_bg_queries = num_bg_queries
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.aux = aux
        self.voxel_size = voxel_size
        self.sample_sizes = sample_sizes

        sizes = self.backbone.PLANES[-5:]

        self.lin_squeeze_head = conv(self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3)

        self.bg_query_feat = nn.Embedding(num_bg_queries, self.mask_dim)
        self.bg_query_pos = nn.Embedding(num_bg_queries, self.mask_dim)

        self.mask_embed_head = nn.Sequential(nn.Linear(self.mask_dim, self.mask_dim), nn.ReLU(), nn.Linear(self.mask_dim, self.mask_dim))

        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_dim, gauss_scale=1.0, normalize=True)

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()

        # Click-to-scene attention
        self.c2s_attention = nn.ModuleList()

        # Click-to-click attention
        self.c2c_attention = nn.ModuleList()

        # FFN
        self.ffn_attention = nn.ModuleList()

        # Scene-to-click attention
        self.s2c_attention = nn.ModuleList()

        num_uniq_decoders = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_uniq_decoders):
            tmp_c2s_attention = nn.ModuleList()
            tmp_s2c_attention = nn.ModuleList()
            tmp_c2c_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            tmp_c2s_attention.append(
                CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_s2c_attention.append(
                CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_c2c_attention.append(
                SelfAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_ffn_attention.append(
                FFNLayer(
                    d_model=self.mask_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            self.c2s_attention.append(tmp_c2s_attention)
            self.s2c_attention.append(tmp_s2c_attention)
            self.c2c_attention.append(tmp_c2c_attention)
            self.ffn_attention.append(tmp_ffn_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.time_encode = PositionalEncoding1D(hidden_dim, 2000)
        self.scan_num_encode = PositionalEncoding1D(hidden_dim, 5000)

        # Learned object-id embeddings
        self.object_embedding = nn.Embedding(400, hidden_dim)
        self.interaction_type_to_id = {"click": 0, "line": 1, "box": 2, "text": 3, "learned_bg": 4}
        self.interaction_type_embedding = nn.Embedding(len(self.interaction_type_to_id), hidden_dim)
        self.interaction_geometry_mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_encoder = TextEncoder(
            output_dim=hidden_dim,
            backend=text_encoder_backend,
            model_name_or_path=text_encoder_model_name_or_path,
            freeze=freeze_text_encoder,
            vocab_size=text_encoder_vocab_size,
            embedding_dim=text_encoder_dim,
        )
        self.text_null_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward_backbone(self, x, raw_coordinates=None, is_eval=False):
        device = x.device
        all_features = self.backbone(x)
        pcd_features = all_features[-1]
        pcd_features = self.lin_squeeze_head(pcd_features)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=all_features[-1].coordinate_manager,
                coordinate_map_key=all_features[-1].coordinate_map_key,
                device=device,
            )
            coords = [coordinates]
            for _ in reversed(range(len(all_features) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)

        return pcd_features, all_features, coordinates, pos_encodings_pcd

    def forward_mask(self, pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=None, click_time_idx=None, scan_numbers=None, interactions=None):

        batch_size = pcd_features.C[:, 0].max() + 1

        predictions_mask = [[] for i in range(batch_size)]

        bg_learn_queries = self.bg_query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        bg_learn_query_pos = self.bg_query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for b in range(batch_size):

            if self.is_cuda_available:
                mins = coordinates.decomposed_features[b].min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.decomposed_features[b].max(dim=0)[0].unsqueeze(0)
            
            else:
                mins = coordinates.F.min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.F.max(dim=0)[0].unsqueeze(0)

            if self.is_cuda_available:
                sample_coords = coordinates.decomposed_features[b]
                sample_features = pcd_features.decomposed_features[b]
            else:
                sample_coords = coordinates.F
                sample_features = pcd_features.F

            interactions_sample = None if interactions is None else interactions[b]
            if interactions_sample is None:
                interactions_sample = self._legacy_clicks_to_interactions(click_idx[b], click_time_idx[b])

            fg_queries, fg_query_pos, fg_query_num_split, bg_queries, bg_query_pos = self._build_unified_interaction_queries(
                interactions_sample=interactions_sample,
                sample_coords=sample_coords,
                sample_features=sample_features,
                mins=mins,
                maxs=maxs,
                bg_learn_queries=bg_learn_queries[b],
                bg_learn_query_pos=bg_learn_query_pos[b],
            )

            fg_query_num = sum(fg_query_num_split)
            bg_query_num = bg_query_pos.shape[0]

            # generate bg (obj 0) obj embedding
            bg_obj_id = torch.zeros(bg_queries.shape[0], dtype=torch.long, device=bg_query_pos.device)
            bg_query_obj = self.object_embedding(bg_obj_id)
            bg_queries += bg_query_obj

            if self.is_cuda_available:
                src_pcd = pcd_features.decomposed_features[b]
            else:
                src_pcd = pcd_features.F

            if self.sweep_size > 1:
                # Add scan encoding for 4d setup for the attention mechanism
                src_pcd_scan_num_encoding = self.scan_num_encode[scan_numbers.cpu().long()].to(src_pcd.device)
                src_pcd += src_pcd_scan_num_encoding

            refine_time = 0

            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                hlevel = 4
                pos_enc = pos_encodings_pcd[hlevel][0][b]  # [num_points, 128]

                if refine_time == 0:
                    attn_mask = None

                output = self.c2s_attention[decoder_counter][0](
                    torch.cat([fg_queries, bg_queries], dim=0),  # [num_queries, 128]
                    src_pcd,  # [num_points, 128]
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,
                    pos=pos_enc,  # [num_points, 128]
                    query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                )  # [num_queries, 128]

                output = self.c2c_attention[decoder_counter][0](
                    output,  # [num_queries, 128]
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                )  # [num_queries, 128]

                # FFN
                queries = self.ffn_attention[decoder_counter][0](output)  # [num_queries, 128]

                src_pcd = self.s2c_attention[decoder_counter][0](
                    src_pcd,
                    queries,  # [num_queries, 128]
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                    query_pos=pos_enc,  # [num_points, 128]
                )  # [num_points, 128]

                fg_queries, bg_queries = queries.split([fg_query_num, bg_query_num], 0)

                outputs_mask, attn_mask = self.mask_module(fg_queries, bg_queries, src_pcd, ret_attn_mask=True, fg_query_num_split=fg_query_num_split)
                predictions_mask[b].append(outputs_mask)

                refine_time += 1

        predictions_mask = [list(i) for i in zip(*predictions_mask)]

        out = {"pred_masks": predictions_mask[-1], "backbone_features": pcd_features}

        if self.aux:
            out["aux_outputs"] = self._set_aux_loss(predictions_mask)

        return out

    def _legacy_clicks_to_interactions(self, click_idx_sample, click_time_idx_sample):
        interactions = {}
        for obj_id, point_indices in click_idx_sample.items():
            interactions[obj_id] = []
            for local_idx, point_idx in enumerate(point_indices):
                times = click_time_idx_sample.get(obj_id, [])
                time_idx = times[local_idx] if local_idx < len(times) else local_idx
                interactions[obj_id].append(
                    {
                        "type": "click",
                        "indices": [int(point_idx)],
                        "time": int(time_idx),
                    }
                )
        return interactions

    def _build_unified_interaction_queries(
        self,
        interactions_sample,
        sample_coords,
        sample_features,
        mins,
        maxs,
        bg_learn_queries,
        bg_learn_query_pos,
    ):
        device = sample_features.device
        fg_queries = []
        fg_query_pos = []
        fg_query_num_split = []

        fg_obj_ids = sorted([int(obj_id) for obj_id in interactions_sample.keys() if int(obj_id) != 0])
        for obj_id in fg_obj_ids:
            obj_tokens = interactions_sample.get(str(obj_id), [])
            obj_query, obj_pos = self._encode_interaction_tokens(
                tokens=obj_tokens,
                obj_id=obj_id,
                sample_coords=sample_coords,
                sample_features=sample_features,
                mins=mins,
                maxs=maxs,
            )
            fg_queries.append(obj_query)
            fg_query_pos.append(obj_pos)
            fg_query_num_split.append(obj_query.shape[0])

        if len(fg_queries) == 0:
            raise ValueError("Interactive4D requires at least one foreground interaction token.")

        fg_queries = torch.cat(fg_queries, dim=0)
        fg_query_pos = torch.cat(fg_query_pos, dim=0)

        bg_tokens = interactions_sample.get("0", [])
        if len(bg_tokens) > 0:
            bg_token_queries, bg_token_pos = self._encode_interaction_tokens(
                tokens=bg_tokens,
                obj_id=0,
                sample_coords=sample_coords,
                sample_features=sample_features,
                mins=mins,
                maxs=maxs,
            )
            bg_queries = torch.cat([bg_learn_queries, bg_token_queries], dim=0)
            bg_query_pos = torch.cat(
                [
                    bg_learn_query_pos + self.interaction_type_embedding.weight[self.interaction_type_to_id["learned_bg"]],
                    bg_token_pos,
                ],
                dim=0,
            )
        else:
            bg_queries = bg_learn_queries
            bg_query_pos = bg_learn_query_pos + self.interaction_type_embedding.weight[self.interaction_type_to_id["learned_bg"]].to(device)

        return fg_queries, fg_query_pos, fg_query_num_split, bg_queries, bg_query_pos

    def _encode_interaction_tokens(self, tokens, obj_id, sample_coords, sample_features, mins, maxs):
        device = sample_features.device
        query_features = []
        query_positions = []

        for token in tokens:
            token_type = token.get("type", "click")
            type_id = self.interaction_type_to_id.get(token_type, self.interaction_type_to_id["click"])
            type_embedding = self.interaction_type_embedding.weight[type_id].to(device)

            indices = token.get("indices", [])
            if len(indices) > 0:
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
                index_tensor = torch.clamp(index_tensor, min=0, max=sample_features.shape[0] - 1)
                token_features = sample_features[index_tensor].mean(dim=0)
                token_coords = sample_coords[index_tensor]
                anchor = token_coords.mean(dim=0)
                extent = token_coords.max(dim=0)[0] - token_coords.min(dim=0)[0]
            else:
                anchor = torch.as_tensor(token.get("anchor", [0.0, 0.0, 0.0]), dtype=sample_coords.dtype, device=device)
                extent = torch.as_tensor(token.get("extent", [0.0, 0.0, 0.0]), dtype=sample_coords.dtype, device=device)
                token_features = torch.zeros(sample_features.shape[1], dtype=sample_features.dtype, device=device)

            if "anchor" in token:
                anchor = torch.as_tensor(token["anchor"], dtype=sample_coords.dtype, device=device)
            if "extent" in token:
                extent = torch.as_tensor(token["extent"], dtype=sample_coords.dtype, device=device)

            token_text = token.get("text", "")
            if token_type == "text" and token_text:
                text_embedding = self.encode_texts([token_text], device=device).squeeze(0)
            else:
                text_embedding = self.text_null_embedding.to(device)
            text_hash = int(token.get("text_hash", 0)) % 4096

            geom = torch.cat(
                [
                    anchor.float(),
                    extent.float(),
                    torch.tensor([float(type_id) / max(1, len(self.interaction_type_to_id) - 1)], device=device),
                    torch.tensor([float(text_hash) / 4096.0], device=device),
                ]
            )
            geom_embedding = self.interaction_geometry_mlp(geom)

            anchor_pos = self.pos_enc(anchor.view(1, 1, 3).float(), input_range=[mins, maxs]).squeeze(0).squeeze(-1)
            time_idx = min(int(token.get("time", 0)), self.time_encode.shape[0] - 1)
            time_embedding = self.time_encode[time_idx].to(device)
            obj_embedding = self.object_embedding(torch.tensor(min(obj_id, self.object_embedding.num_embeddings - 1), dtype=torch.long, device=device))

            query_features.append(token_features + type_embedding + geom_embedding + text_embedding)
            query_positions.append(anchor_pos + time_embedding + obj_embedding + type_embedding + geom_embedding + text_embedding)

        if len(query_features) == 0:
            raise ValueError(f"Object {obj_id} has no interaction tokens.")

        return torch.stack(query_features, dim=0), torch.stack(query_positions, dim=0)

    def encode_texts(self, texts, device=None, normalize=False):
        device = device or self.text_null_embedding.device
        if normalize:
            return self.text_encoder.normalized(texts, device=device)
        return self.text_encoder(texts, device=device)

    def mask_module(self, fg_query_feat, bg_query_feat, mask_features, ret_attn_mask=True, fg_query_num_split=None):

        fg_query_feat = self.decoder_norm(fg_query_feat)
        fg_mask_embed = self.mask_embed_head(fg_query_feat)

        fg_prods = mask_features @ fg_mask_embed.T
        fg_prods = fg_prods.split(fg_query_num_split, dim=1)

        fg_masks = []
        for fg_prod in fg_prods:
            fg_masks.append(fg_prod.max(dim=-1, keepdim=True)[0])

        fg_masks = torch.cat(fg_masks, dim=-1)

        bg_query_feat = self.decoder_norm(bg_query_feat)
        bg_mask_embed = self.mask_embed_head(bg_query_feat)
        bg_masks = (mask_features @ bg_mask_embed.T).max(dim=-1, keepdim=True)[0]

        output_masks = torch.cat([bg_masks, fg_masks], dim=-1)

        if ret_attn_mask:

            output_labels = output_masks.argmax(1)

            bg_attn_mask = ~(output_labels == 0)  # Masking all the points which were *not* predicted as background
            bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
            # prevent a scenario where a query would be completely masked out and have nothing
            # to attend to, which could cause issues in the attention mechanism.
            bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False

            fg_attn_mask = []
            for fg_obj_id in range(1, fg_masks.shape[-1] + 1):
                fg_obj_mask = ~(output_labels == fg_obj_id)  # Masking all the points which were *not* predicted as the obj_id
                fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id - 1], 1)
                # prevent a scenario where a query would be completely masked out and have nothing
                # to attend to, which could cause issues in the attention mechanism.
                fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
                fg_attn_mask.append(fg_obj_mask)

            fg_attn_mask = torch.cat(fg_attn_mask, dim=0)

            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)

            return output_masks, attn_mask

        return output_masks

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            if self.is_cuda_available:
                decomposed_features = coords[i].decomposed_features
            else:
                decomposed_features = [coords[i].F]
            for coords_batch in decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(), input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def get_random_samples(self, pcd_sizes, curr_sample_size, device):
        rand_idx = []
        mask_idx = []
        for pcd_size in pcd_sizes:
            if pcd_size <= curr_sample_size:
                # we do not need to sample
                # take all points and pad the rest with zeroes and mask it
                idx = torch.zeros(curr_sample_size, dtype=torch.long, device=device)
                midx = torch.ones(curr_sample_size, dtype=torch.bool, device=device)
                idx[:pcd_size] = torch.arange(pcd_size, device=device)
                midx[:pcd_size] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(pcd_size, device=device)[:curr_sample_size]
                midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=device)

            rand_idx.append(idx)
            mask_idx.append(midx)
        return rand_idx, mask_idx

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [{"pred_masks": a} for a in outputs_seg_masks[:-1]]
