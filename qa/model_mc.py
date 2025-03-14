import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMultipleChoice

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class QACRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        for i in range(args.n_model):
            model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, no_nll=True):
        """
            input_ids: [batch_size, num_choices, max_len]
            attention_mask: [batch_size, num_choices, max_len]
            token_type_ids: [batch_size, num_choices, max_len]
            labels: (batch_size, )
        """

        if labels is None:
            return self.models[0](input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  return_dict=False,
                                  )
        else:
            num_models = len(self.models)
            outputs = [] # outputs of different models in model_list
            for i in range(num_models):
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]),
                    return_dict=False,
                )
                output = tuple([o.to(0) for o in output])
                outputs.append(output)
            model_output = outputs[0] # the output of the first model.

            cr_loss = sum([output[0] for output in outputs]) / num_models # average the loss
            if no_nll: # no nll loss
                # only the loss is needed
                return (cr_loss, [output[1] for output in outputs]) # loss and logits
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            mask = (labels.view(-1) != -1).to(logits[0])
            reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / num_models
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
            loss = cr_loss + self.args.alpha_t * reg_loss # new loss
            # model_output = (loss,) + model_output[1:]
            model_output = (loss, logits, cr_loss, reg_loss)
        return model_output