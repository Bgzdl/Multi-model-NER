import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class TextEncoder(nn.Module):
    def __init__(
            self,
            base_model,
            proj_dim,
            use_act_fn=False,
            frozen_base=True
    ):
        super().__init__()

        self.base_model = base_model
        if frozen_base:
            for param in base_model.parameters():
                param.requires_grad = False

        input_dim = base_model.config.hidden_size
        print(base_model.config.hidden_size)
        self.projection = nn.Linear(input_dim, proj_dim)
        self.act_fn = nn.ReLU() if use_act_fn else nn.Identity()

    def forward(self, token):
        output = self.base_model(**token)[0]

        proj_out = self.projection(output)
        proj_out = self.act_fn(proj_out)

        return proj_out


def build_text_encoder(
        model_name='bert-base-cased',
        frozen=False,
        proj_dim=768
):
    """
        Load BERT model and return the Text encoder and tokenizer.
        :param model_name: available models on Huggingface:
        ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased']
        :param frozen: if the base model is frozen or not
        :param proj_dim: the dimension of the output projection
        :return: bert model with projection and its tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    text_encoder = TextEncoder(model, proj_dim, frozen_base=frozen)

    return tokenizer, text_encoder


if __name__ == '__main__':
    tokenizer, model = build_text_encoder()

    text = ["Hello, how are you doing?", "I'm doing fine, thank you!"]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    print(encoded_input)
    output = model(encoded_input)

    print(output.shape)
