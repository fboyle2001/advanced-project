
import torch
# import models.prompt.prompt_vit as m

# vit = m.vit_b_16(weights=m.ViT_B_16_Weights.DEFAULT)
# # print(fe.get_graph_node_names(vit))

# img = torch.randn(1, 3, 224, 224)

# # This returns batch x n_patches x C * H_patch * W_patch i.e. 1 x 196 x 768 for a 1 x 3 x 224 x 224
# # i.e. reshapes the input as required
# x_p = vit._process_input(img)
# print("x_p", x_p.shape)
# # test = torch.randn(1, 4, 768)
# # x = torch.concat([x_p, test], dim=1)
# # print("x_", x_p.shape)

# # This gets the class token
# n = x_p.shape[0]
# # # Expand the class token to the full batch
# batch_class_token = vit.class_token.expand(n, -1, -1)
# x = torch.cat([batch_class_token, x_p], dim=1)
# # This returns batch x L x (S^2 * C) i.e. this will give x_p 
# x = vit.encoder(x)
# print("enc", x.shape)
# print(x[:, 0].shape)

from vit.vit_models import create_model

img = torch.randn(1, 3, 224, 224)

vit = create_model()
# print(type(vit))
# print(type(vit.enc))
# print(vit.enc.transformer(img)[0].shape)

# Embeddings then encoder

embeddings = vit.enc.transformer.embeddings(img)
print(embeddings.shape)
# At this point, incorporate the prompts
# Like this since we have to have class token at index 0, prompts, remaining
# x = torch.cat((
#                 x[:, :1, :],
#                 self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
#                 x[:, 1:, :]
#             ), dim=1)
prompts = torch.randn(4, 5, 768)
print(prompts.shape)
# Has class token at the front
prompt_prepended_embeddings = torch.cat([embeddings[:, :1, :], prompts, embeddings[:, 1:, :]], dim=1)
print(prompt_prepended_embeddings.shape)

encoded, attn_weights = vit.enc.transformer.encoder(prompt_prepended_embeddings)
p = encoded[:, 0:6]
print(p.shape)

pool = torch.nn.AvgPool2d(kernel_size=6)
pooled = pool(p).squeeze()
print(pooled.shape)

classifier = torch.nn.Linear(in_features=128, out_features=10)
c = classifier(pooled)
print(c, c.shape)


# torch._assert(not torch.all(torch.eq(prompt_prepended_embeddings, encoded)), "Equal")
# print("Here")
# x, _ = vit.enc.transformer(img)
# x = x[:, 0]
# print(x.shape)

# print("Features:", x.shape)