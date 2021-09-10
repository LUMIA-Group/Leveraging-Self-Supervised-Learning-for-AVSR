import os
import shutil

import fairseq
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.plugins import *
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_warmup_lr import WarmupLR

from config import args
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.moco_visual_frontend import MoCoVisualFrontend
from models.utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask
from utils.decoders import ctc_greedy_decode, teacher_forcing_attention_decode
from utils.label_smoothing import SmoothCTCLoss, SmoothCrossEntropyLoss
from utils.metrics import compute_error_ch, compute_error_word


class LRS2Lightning(pl.LightningDataModule):
    def __init__(self):
        super(LRS2Lightning, self).__init__()
        self.kwargs = {"num_workers": args["NUM_WORKERS"], "persistent_workers": True if args["NUM_WORKERS"] > 0 else False, "pin_memory": True}

    def setup(self, stage):
        if stage == "fit" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": args["NOISE_PROBABILITY"], "noiseSNR": args["NOISE_SNR_DB"]}
            self.trainData = LRS2(args['MODAL'], "train", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                  True, noiseParams)

            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.valData = LRS2(args['MODAL'], "val", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                noiseParams)

        if stage == "test" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.testData = LRS2(args['MODAL'], "test", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                 noiseParams)

    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)


class AVNet(pl.LightningModule):

    def __init__(self, modal, W2Vfile, MoCofile, reqInpLen, modelargs):
        super(AVNet, self).__init__()

        self.trainParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "modal": args["MODAL"],
                            "Alpha": args["ALPHA"]}

        self.valParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "modal": args["MODAL"],
                          "Alpha": args["ALPHA"]}

        self.ft = False

        self.CTCLossFunction = [SmoothCTCLoss(args["CHAR_NUM_CLASSES"], blank=0)]
        self.CELossFunction = [SmoothCrossEntropyLoss()]

        dModel, nHeads, numLayers, peMaxLen, audinSize, vidinSize, fcHiddenSize, dropout, numClasses = modelargs
        self.modal = modal
        self.numClasses = numClasses
        self.reqInpLen = reqInpLen
        # A & V Modal
        tx_norm = nn.LayerNorm(dModel)
        self.maskedLayerNorm = MaskedLayerNorm()
        if self.modal == "AV":
            self.ModalityNormalization = nn.LayerNorm(dModel)
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        # audio
        if not self.modal == "VO":
            # front-end
            wav2vecModel, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([W2Vfile], arg_overrides={
                "apply_mask": True,
                "mask_prob": 0.5,
                "mask_channel_prob": 0.25,
                "mask_channel_length": 64,
                "layerdrop": 0.1,
                "activation_dropout": 0.1,
                "feature_grad_mult": 0.0,
            })
            wav2vecModel = wav2vecModel[0]
            wav2vecModel.remove_pretraining_modules()
            self.wav2vecModel = wav2vecModel
            # back-end
            self.audioConv = conv1dLayers(self.maskedLayerNorm, audinSize, dModel, dModel, downsample=True)
            audioEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.audioEncoder = nn.TransformerEncoder(audioEncoderLayer, num_layers=numLayers, norm=tx_norm)
        else:
            self.wav2vecModel = None
            self.audioConv = None
            self.audioEncoder = None
        # visual
        if not self.modal == "AO":
            # front-end
            visualModel = MoCoVisualFrontend()
            visualModel.load_state_dict(torch.load(MoCofile, map_location="cpu"), strict=False)
            self.visualModel = visualModel
            # back-end
            self.videoConv = conv1dLayers(self.maskedLayerNorm, vidinSize, dModel, dModel)
            videoEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.videoEncoder = nn.TransformerEncoder(videoEncoderLayer, num_layers=numLayers, norm=tx_norm)
        else:
            self.visualModel = None
            self.videoConv = None
            self.videoEncoder = None
        # JointConv for fusion
        if self.modal == "AV":
            self.jointConv = conv1dLayers(self.maskedLayerNorm, 2 * dModel, dModel, dModel)
            jointEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.jointEncoder = nn.TransformerEncoder(jointEncoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointOutputConv = outputConv(self.maskedLayerNorm, dModel, numClasses)
        self.decoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.embed = torch.nn.Sequential(
            nn.Embedding(numClasses, dModel),
            self.decoderPositionalEncoding
        )
        jointDecoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointAttentionOutputConv = outputConv("LN", dModel, numClasses)
        return

    def subNetForward(self, inputBatch, maskw2v):
        audioBatch, audMask, videoBatch, vidLen = inputBatch
        if not self.modal == "VO":
            if self.ft or not self.modal == "AV":
                audioBatch, audMask = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)
            else:
                with torch.no_grad():
                    audioBatch, audMask = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)

            audLen = torch.sum(~audMask, dim=1)
        else:
            audLen = None

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            if self.modal == "AV":
                with torch.no_grad():
                    videoBatch = self.visualModel(videoBatch, vidLen.long())
            else:
                videoBatch = self.visualModel(videoBatch, vidLen.long())

            videoBatch = list(torch.split(videoBatch, vidLen.tolist(), dim=0))

        audioBatch, videoBatch, inputLenBatch, mask = self.makePadding(audioBatch, audLen, videoBatch, vidLen)

        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        if not self.modal == "VO":
            if self.modal == "AV":
                with torch.no_grad():
                    audioBatch = audioBatch.transpose(1, 2)
                    audioBatch = self.audioConv(audioBatch)
                    audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
                    audioBatch = self.EncoderPositionalEncoding(audioBatch)
                    audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)
            else:
                audioBatch = audioBatch.transpose(1, 2)
                audioBatch = self.audioConv(audioBatch)
                audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
                audioBatch = self.EncoderPositionalEncoding(audioBatch)
                audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)

        if not self.modal == "AO":
            if self.modal == "AV":
                with torch.no_grad():
                    videoBatch = videoBatch.transpose(1, 2)
                    videoBatch = self.videoConv(videoBatch)
                    videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
                    videoBatch = self.EncoderPositionalEncoding(videoBatch)
                    videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask)
            else:
                videoBatch = videoBatch.transpose(1, 2)
                videoBatch = self.videoConv(videoBatch)
                videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
                videoBatch = self.EncoderPositionalEncoding(videoBatch)
                videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask)

        if self.modal == "AO":
            jointBatch = audioBatch
        elif self.modal == "VO":
            jointBatch = videoBatch
        else:
            jointBatch = torch.cat([self.ModalityNormalization(audioBatch), self.ModalityNormalization(videoBatch)], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            jointBatch = self.EncoderPositionalEncoding(jointBatch)
            jointBatch = self.jointEncoder(jointBatch, src_key_padding_mask=mask)

        return jointBatch, inputLenBatch, mask

    def forward(self, inputBatch, targetinBatch, targetLenBatch, maskw2v):
        jointBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v)
        jointCTCOutputBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointCTCOutputBatch = self.jointOutputConv(jointCTCOutputBatch)
        jointCTCOutputBatch = jointCTCOutputBatch.transpose(1, 2).transpose(0, 1)
        jointCTCOutputBatch = F.log_softmax(jointCTCOutputBatch, dim=2)

        targetinBatch = self.embed(targetinBatch.transpose(0, 1))
        targetinMask = self.makeMaskfromLength(targetinBatch.shape[:-1][::-1], targetLenBatch, self.device)
        squareMask = generate_square_subsequent_mask(targetinBatch.shape[0], self.device)
        jointAttentionOutputBatch = self.jointAttentionDecoder(targetinBatch, jointBatch, tgt_mask=squareMask,
                                                               tgt_key_padding_mask=targetinMask, memory_key_padding_mask=mask)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(0, 1).transpose(1, 2)
        jointAttentionOutputBatch = self.jointAttentionOutputConv(jointAttentionOutputBatch)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(1, 2)

        outputBatch = (jointCTCOutputBatch, jointAttentionOutputBatch)
        return inputLenBatch, outputBatch

    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def makePadding(self, audioBatch, audLen, videoBatch, vidLen):
        if self.modal == "AO":
            audPadding = audLen % 2
            mask = (audPadding + audLen) > 2 * self.reqInpLen
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
            audioBatch = list(audioBatch)
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            inputLenBatch = ((audLen + audPadding) // 2).long()
            mask = self.makeMaskfromLength([audioBatch.shape[0]] + [audioBatch.shape[1] // 2], inputLenBatch, self.device)

        elif self.modal == "VO":
            vidPadding = torch.zeros(len(videoBatch)).long().to(self.device)

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()

            for i, _ in enumerate(videoBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, self.device)

        else:
            dismatch = audLen - 2 * vidLen
            vidPadding = torch.ceil(torch.div(dismatch, 2)).int()
            vidPadding = vidPadding * (vidPadding > 0)
            audPadding = 2 * vidPadding - dismatch

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)
            mask = (audPadding + audLen) > 2 * self.reqInpLen
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
            audioBatch = list(audioBatch)
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, self.device)

        return audioBatch, videoBatch, inputLenBatch, mask

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
        scheduler_reduce = ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"], patience=args["LR_SCHEDULER_WAIT"],
                                             threshold=args["LR_SCHEDULER_THRESH"], threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
        if args["LRW_WARMUP_PERIOD"] > 0:
            scheduler = WarmupLR(scheduler_reduce, init_lr=args["FINAL_LR"], num_warmup=args["LRS2_WARMUP_PERIOD"], warmup_strategy='cos')
            scheduler.step(1)
        else:
            scheduler = scheduler_reduce

        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'epoch',  # The unit of the scheduler's step size
                'frequency': 1,  # The frequency of the scheduler
                'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                'monitor': 'CER/val_CER',
                'strict': True,  # Whether to crash the training if `monitor` is not found
                'name': None,  # Custom name for LearningRateMonitor to use
            }
        }
        return optim_dict

    def training_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch
        Alpha = self.trainParams["Alpha"]

        if self.trainParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.trainParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]

        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), True)
        with torch.backends.cudnn.flags(enabled=False):
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch)
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss
        self.log("info/train_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/train_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0].detach(), inputLenBatch, self.trainParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/train_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.trainParams["spaceIx"])
        self.log("info/train_WER", w_edits / w_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch
        Alpha = self.trainParams["Alpha"]

        if self.valParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.valParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]

        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), False)
        with torch.backends.cudnn.flags(enabled=False):
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch)
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss
        self.log("info/val_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/val_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch, self.valParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], self.valParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_TF_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_TF_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class UnfreezeCallback(Callback):
    """Unfreeze feature extractor callback."""

    def on_epoch_start(self, trainer, pl_module):
        if not pl_module.ft and trainer.current_epoch > args["W2V_FREEZE_EPOCH"]:
            pl_module.ft = True

    def on_train_epoch_start(self, trainer, pl_module):
        if not pl_module.ft:
            pl_module.wav2vecModel.eval()
        if args["MODAL"] == "AV":
            pl_module.wav2vecModel.eval()
            pl_module.audioConv.eval()
            pl_module.audioEncoder.eval()
            pl_module.visualModel.eval()
            pl_module.videoConv.eval()
            pl_module.videoEncoder.eval()


def main():
    pl.seed_everything(args["SEED"])
    torch.set_num_threads(args["NUM_CPU_CORE"])
    LRS2Dataloader = LRS2Lightning()
    LRS2Dataloader.setup('fit')
    modelargs = (args["DMODEL"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"],
                 args["VIDEO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["CHAR_NUM_CLASSES"])
    model = AVNet(args['MODAL'], args['WAV2VEC_FILE'], args['MOCO_FRONTEND_FILE'], args["MAIN_REQ_INPUT_LENGTH"], modelargs)
    # loading the pretrained weights
    if not args["MODAL"] == "AV" and args["TRAIN_LRS2_MODEL_FILE"] is not None:
        stateDict = torch.load(args["TRAIN_LRS2_MODEL_FILE"], map_location="cpu")['state_dict']
        model.load_state_dict(stateDict, strict=False)
    elif args["TRAINED_AO_FILE"] is not None and args["TRAINED_VO_FILE"] is not None:
        AOstateDict = torch.load(args["TRAINED_AO_FILE"])['state_dict']
        stateDict = torch.load(args["TRAINED_VO_FILE"])['state_dict']
        for k in list(AOstateDict.keys()):
            if not (k.startswith('audioConv') or k.startswith('wav2vecModel')):
                del AOstateDict[k]

        for k in list(stateDict.keys()):
            if not (k.startswith('videoConv') or k.startswith('visualModel')):
                del stateDict[k]
        stateDict.update(AOstateDict)
        model.load_state_dict(stateDict, strict=False)

    writer = pl_loggers.TensorBoardLogger(save_dir=args["CODE_DIRECTORY"], name='log', default_hp_metric=False)
    # removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "checkpoints"):
        shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["CODE_DIRECTORY"] + "checkpoints/models",
        filename=
        "train-step_{epoch:04d}-cer_{CER/val_CER:.3f}" if args["LR_SCHEDULER_METRICS"] == "CER" else "train-step_{epoch:04d}-wer_{info/val_WER:.3f}",
        monitor='CER/val_CER' if args["LR_SCHEDULER_METRICS"] == "CER" else 'info/val_WER',
        every_n_epochs=1,
        every_n_train_steps=0,
        save_top_k=20,
        mode="min",
        auto_insert_metric_name=False,
        save_weights_only=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args["W2V_FREEZE_EPOCH"] > 0:
        callback_list = [checkpoint_callback, lr_monitor, UnfreezeCallback()]
    else:
        callback_list = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        gpus=args["GPU_IDS"],
        benchmark=False,
        deterministic=True,
        logger=writer,
        default_root_dir=args["CODE_DIRECTORY"],
        callbacks=callback_list,
        accelerator="dp",
        plugins=DDPPlugin(find_unused_parameters=False if args["MODAL"] == "VO" else True),
    )
    trainer.fit(model, LRS2Dataloader)
    return


if __name__ == "__main__":
    main()
