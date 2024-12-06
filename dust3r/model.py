# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')  # 定义无穷大

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")  # 确保 huggingface_hub 版本符合要求


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)  # 打印加载模型的路径
    ckpt = torch.load(model_path, map_location='cpu')  # 从指定路径加载模型
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")  # 替换模型参数
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'  # 如果没有 landscape_only 参数，添加默认值
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')  # 替换为 False
    assert "landscape_only=False" in args  # 确保参数设置正确
    if verbose:
        print(f"instantiating : {args}")  # 打印实例化的参数
    net = eval(args)  # 动态实例化模型
    s = net.load_state_dict(ckpt['model'], strict=False)  # 加载模型状态字典
    if verbose:
        print(s)  # 打印加载状态
    return net.to(device)  # 将模型移动到指定设备


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """  # 两个相似的编码器，后跟两个解码器，目标是直接输出3D点���两个图像在视图1的框架中

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R 或 ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)  # 填充默认参数
        super().__init__(**croco_kwargs)  # 初始化父类

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)  # 深拷贝解码块
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)  # 设置下游头
        self.set_freeze(freeze)  # 设置冻结参数

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')  # 从文件加载预训练模型
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)  # 从 huggingface 加载模型
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')  # 抛出加载失败的异常
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)  # 设置图像补丁嵌入

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)  # 创建状态字典的副本
        if not any(k.startswith('dec_blocks2') for k in ckpt):  # 检查是否存在第二个解码器的权重
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value  # 复制权重
        return super().load_state_dict(new_ckpt, **kw)  # 加载状态字典

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze  # 设置冻结状态
        to_be_frozen = {
            'none': [],  # 不冻结
            'mask': [self.mask_token],  # 冻结掩码
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],  # 冻结编码器
        }
        freeze_all_params(to_be_frozen[freeze])  # 冻结参数

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """  # 不设置预测头
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'  # 确保图像尺寸是补丁大小的倍数
        self.output_mode = output_mode  # 设置输出模式
        self.head_type = head_type  # 设置头类型
        self.depth_mode = depth_mode  # 设置深度模式
        self.conf_mode = conf_mode  # 设置置信度模式
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))  # 创建下游头1
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))  # 创建下游头2
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)  # 转换为横向布局
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)  # 转换为横向布局

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)  # 将图像嵌入为补丁
        x, pos = self.patch_embed(image, true_shape=true_shape)  # 获取补丁和位置嵌入

        # add positional embedding without cls token
        assert self.enc_pos_embed is None  # 确保没有位置嵌入

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)  # 应用编码器块

        x = self.enc_norm(x)  # 归一化
        return x, pos, None  # 返回编码后的特征和位置

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:  # 检查图像尺寸是否相同
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))  # 编码图像对
            out, out2 = out.chunk(2, dim=0)  # 分割输出
            pos, pos2 = pos.chunk(2, dim=0)  # 分割位置
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)  # 编码图像1
            out2, pos2, _ = self._encode_image(img2, true_shape2)  # 编码图像2
        return out, out2, pos, pos2  # 返回编码结果

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']  # 获取视图1的图像
        img2 = view2['img']  # 获取视图2的图像
        B = img1.shape[0]  # 获取批次大小
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))  # 获取真实形状
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))  # 获取真实形状
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):  # 检查是否对称
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])  # 编码图像对
            feat1, feat2 = interleave(feat1, feat2)  # 交错特征
            pos1, pos2 = interleave(pos1, pos2)  # 交错位置
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)  # 编码图像对

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)  # 返回形状和特征

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection  # 投影前的输出

        # project to decoder dim
        f1 = self.decoder_embed(f1)  # 嵌入解码器
        f2 = self.decoder_embed(f2)  # 嵌入解码器

        final_output.append((f1, f2))  # 添加到输出
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):  # 遍历解码块
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)  # 解码图像1
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)  # 解码图像2
            # store the result
            final_output.append((f1, f2))  # 存储结果

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]  # 删除重复的输出
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))  # 归一化最后的输出
        return zip(*final_output)  # 返回解码结果

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape  # 获取输出形状
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')  # 获取对应的头
        return head(decout, img_shape)  # 返回头的输出

    def forward(self, view1, view2):
        # encode the two images --> B,S,D  # 编码两个图像，输出形状为 B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)  # 编码对称图像

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)  # 解码特征

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)  # 获取第一个头的输出
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)  # 获取第二个头的输出

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame  # 在视图1的框架中预测视图2的3D点
        return res1, res2  # 返回两个视图的结果
