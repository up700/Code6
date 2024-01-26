# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig, RobertaModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
from utils.loss_utils import EndoLoss, OrthogonalLoss

class Task_Decom(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, type_num_labels, lambda_0, num_epochs, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        self.type_num_labels = type_num_labels+1

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

        self.classifier_sp = nn.Linear(config.hidden_size, self.span_num_labels)
        self.classifier_tp = nn.Linear(config.hidden_size, self.type_num_labels)

        self.shared_feat = nn.Linear(config.hidden_size, config.hidden_size)
        self.span_feat = nn.Linear(config.hidden_size, config.hidden_size)
        self.type_feat = nn.Linear(config.hidden_size, config.hidden_size)

        self.endo_sp_loss = EndoLoss(config.hidden_size, self.span_num_labels)
        self.endo_tp_loss = EndoLoss(config.hidden_size, self.type_num_labels)

        self.coef = lambda_0/num_epochs*1.0

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_sp=None,
        labels_tp=None,
        entity_mask=None,
        input_ids_ctx=None,
        epoch=0,
        beta1=0.1,
        beta2=0.1
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        final_embedding = outputs[0] # B, L, D
        sequence_output = self.dropout(final_embedding)
        
        feat_shared = self.shared_feat(sequence_output)
        feat_task = sequence_output-feat_shared
        feat_sp = self.span_feat(feat_task)
        feat_tp = self.type_feat(feat_task)

        loss_orth = OrthogonalLoss()

        seq_sp = feat_shared + feat_sp
        seq_tp = feat_shared + feat_tp
        logits_sp = self.classifier_sp(seq_sp) # B, L, C_span
        logits_tp = self.classifier_tp(seq_tp) # B, L, C_type


        if entity_mask is not None:
            entity_mask = entity_mask.float()

            att_mask = torch.bmm(entity_mask.unsqueeze(-1), entity_mask.unsqueeze(1)) # B*L*L

            outputs_ent = self.roberta(
                input_ids,
                attention_mask=att_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            )
            ent_embedding = outputs_ent[0] # B, L, D
            sequence_output_ent = self.dropout(ent_embedding)
            feat_shared_ent = self.shared_feat(sequence_output_ent)
            feat_task_ent = sequence_output_ent-feat_shared_ent
            feat_sp_ent = self.span_feat(feat_task_ent)
            feat_tp_ent = self.type_feat(feat_task_ent)

            seq_sp_ent = feat_shared_ent + feat_sp_ent
            seq_tp_ent = feat_shared_ent + feat_tp_ent
            logits_sp_ent = self.classifier_sp(seq_sp_ent) # B, L, C_span
            logits_tp_ent = self.classifier_tp(seq_tp_ent) # B, L, C_type

            outputs_ctx = self.roberta(
                input_ids_ctx,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            )
            ctx_embedding = outputs_ctx[0] # B, L, D
            sequence_output_ctx = self.dropout(ctx_embedding)
            feat_shared_ctx = self.shared_feat(sequence_output_ctx)
            feat_task_ctx = sequence_output_ctx-feat_shared_ctx
            feat_sp_ctx = self.span_feat(feat_task_ctx)
            feat_tp_ctx = self.type_feat(feat_task_ctx)

            seq_sp_ctx = feat_shared_ctx + feat_sp_ctx
            seq_tp_ctx = feat_shared_ctx + feat_tp_ctx
            logits_sp_ctx = self.classifier_sp(seq_sp_ctx) # B, L, C_span
            logits_tp_ctx = self.classifier_tp(seq_tp_ctx) # B, L, C_type


        
        outputs = (logits_sp, logits_tp, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        if labels_sp is not None:
            loss_reg = loss_orth(feat_shared, feat_sp) + loss_orth(feat_shared, feat_tp) + loss_orth(feat_sp, feat_tp)
            
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss_sp = True
            active_loss_tp = True
            if attention_mask is not None:
                active_loss_sp = (attention_mask.view(-1) == 1)&(labels_sp.view(-1)>=0)
                active_loss_tp = (attention_mask.view(-1) == 1)&(labels_tp.view(-1)>=0)
            
            active_logits_sp = logits_sp.view(-1, self.span_num_labels)[active_loss_sp]
            active_logits_tp = logits_tp.view(-1, self.type_num_labels)[active_loss_tp]
            active_sequence_output_sp = seq_sp.view(-1, seq_sp.size(-1))[active_loss_sp]
            active_sequence_output_tp = seq_tp.view(-1, seq_tp.size(-1))[active_loss_tp]

            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_labels_sp = labels_sp.view(-1)[active_loss_sp]
                active_labels_tp = labels_tp.view(-1)[active_loss_tp]
                # loss_bio = loss_fct(active_logits, active_labels)
            else:
                active_labels_sp = labels_sp.view(-1)
                active_labels_tp = labels_tp.view(-1)


            # self, fc, logits, features, target_x, ratio, logits_ent, features_ent, logits_ctx, features_ctx, entity_mask
            if entity_mask is not None:
                loss_reg += loss_orth(feat_shared_ent, feat_sp_ent) + loss_orth(feat_shared_ent, feat_tp_ent) + loss_orth(feat_sp_ent, feat_tp_ent)
                loss_reg += loss_orth(feat_shared_ctx, feat_sp_ctx) + loss_orth(feat_shared_ctx, feat_tp_ctx) + loss_orth(feat_sp_ctx, feat_tp_ctx)

                active_logits_sp_ent = logits_sp_ent.view(-1, self.span_num_labels)[active_loss_sp]
                active_logits_tp_ent = logits_tp_ent.view(-1, self.type_num_labels)[active_loss_tp]
                active_sequence_output_sp_ent = seq_sp_ent.view(-1, seq_sp_ent.size(-1))[active_loss_sp]
                active_sequence_output_tp_ent = seq_tp_ent.view(-1, seq_tp_ent.size(-1))[active_loss_tp]

                active_logits_sp_ctx = logits_sp_ctx.view(-1, self.span_num_labels)[active_loss_sp]
                active_logits_tp_ctx = logits_tp_ctx.view(-1, self.type_num_labels)[active_loss_tp]
                active_sequence_output_sp_ctx = seq_sp_ctx.view(-1, seq_sp_ctx.size(-1))[active_loss_sp]
                active_sequence_output_tp_ctx = seq_tp_ctx.view(-1, seq_tp_ctx.size(-1))[active_loss_tp]

                entity_mask = entity_mask.view(-1)[active_loss_sp]
                loss_sp = self.endo_sp_loss(self.classifier_sp, active_logits_sp, active_sequence_output_sp, active_labels_sp, ratio=epoch*self.coef,\
                                            logits_ent=active_logits_sp_ent, features_ent=active_sequence_output_sp_ent, logits_ctx=active_logits_sp_ctx,\
                                            features_ctx=active_sequence_output_sp_ctx, entity_mask=entity_mask, beta1=beta1, beta2=beta2)
                loss_tp = self.endo_tp_loss(self.classifier_tp, active_logits_tp, active_sequence_output_tp, active_labels_tp, ratio=epoch*self.coef,\
                                            logits_ent=active_logits_tp_ent, features_ent=active_sequence_output_tp_ent, logits_ctx=active_logits_tp_ctx,\
                                            features_ctx=active_sequence_output_tp_ctx, entity_mask=entity_mask, beta1=beta1, beta2=beta2)
            else:
                loss_sp = self.endo_sp_loss(self.classifier_sp, active_logits_sp, active_sequence_output_sp, active_labels_sp, ratio=epoch*self.coef, beta1=beta1, beta2=beta2)
                loss_tp = self.endo_tp_loss(self.classifier_tp, active_logits_tp, active_sequence_output_tp, active_labels_tp, ratio=epoch*self.coef, beta1=beta1, beta2=beta2)



            outputs = (loss_sp, loss_tp, loss_reg, active_logits_sp, active_logits_tp,) + outputs

        return outputs
