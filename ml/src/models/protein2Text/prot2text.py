import torch
from torch import nn
from transformers import AutoModelForCausalLM
from ml.src.models.common.normalization.rms import RMSNorm
from ml.src.models.common.activation_functions.silu import SILU

class Prot2Text(nn.Module):
    def __init__(self, protein_model, llm_model, input_dim, hidden_dim, output_dim, dropout=0.1, bias=False):
        super().__init__()
        self.protein_model = protein_model
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            SILU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )
        self.norm = RMSNorm(output_dim)
        self.llm_model = llm_model

    def forward(self, protein_ids, protein_attention_mask, text_idxs, attention_mask, labels):
        protein_output = self.protein_model(input_ids=protein_ids, attention_mask=protein_attention_mask, output_hidden_states=True)
        protein_hidden = protein_output.hidden_states[-1]
        protein_memory = self.projection(protein_hidden)
        protein_memory = self.norm(protein_memory)
        output = self.llm_model(input_ids=text_idxs,
                                          attention_mask=attention_mask,
                                            encoder_hidden_states=protein_memory,
                                                encoder_attention_mask=protein_attention_mask,
                                                    labels=labels)
        return output

    @torch.no_grad()
    def generate(
        self,
        protein_ids,
        protein_attention_mask,
        max_new_tokens=64,
    ):
        self.eval()

        protein_output = self.protein_model(
            input_ids=protein_ids,
            attention_mask=protein_attention_mask,
            output_hidden_states=True,
        )

        protein_hidden = protein_output.hidden_states[-1]

        protein_memory = self.projection(protein_hidden)
        protein_memory = self.norm(protein_memory)

        bos_token_id = self.llm_model.config.bos_token_id

        if bos_token_id is None:
            bos_token_id = self.llm_model.config.eos_token_id

        text_ids = torch.full(
            (protein_ids.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=protein_ids.device,
        )


        generated_ids = self.llm_model.generate(
            input_ids=text_ids,
            encoder_hidden_states=protein_memory,
            encoder_attention_mask=protein_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm_model.config.eos_token_id,
            eos_token_id=self.llm_model.config.eos_token_id,
        )

        return generated_ids