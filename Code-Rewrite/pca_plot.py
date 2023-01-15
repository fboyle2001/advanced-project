from typing import Dict, List, Tuple

from sklearn.decomposition import PCA
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision
from PIL import Image

from models.vit import vit_models

base_cifar_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) 
])

device = "cuda:0"
pretrained_vit = None

def convert_samples_to_vit_features(samples: List[np.ndarray]) -> List[np.ndarray]:
    global pretrained_vit

    if pretrained_vit is None:
        pretrained_vit = vit_models.create_model_non_prompt(len(samples)).to(device).eval()
    
    s = [base_cifar_transform(Image.fromarray(sample.astype(np.uint8))) for sample in samples]
    tensor_samples = torch.stack(s).to(device) # type: ignore
    embeddings = pretrained_vit.enc.transformer.embeddings(tensor_samples)
    encoded, _ = pretrained_vit.enc.transformer.encoder(embeddings)
    encoded = encoded[:, 0, :]
    vit_features = encoded / torch.linalg.norm(encoded, dim=1).unsqueeze(1)
    vit_features = vit_features.cpu().numpy()

    return [x for x in vit_features]

def load_samples_and_means(file_name: str) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, np.ndarray]]:
    stored = None

    with open(file_name, "r") as fp:
        stored = json.load(fp)
    
    assert stored is not None

    memory: Dict[int, List[np.ndarray]] = {}

    for target in stored["memory"].keys():
        memory[int(target)] = [np.asarray(sample) for sample in stored["memory"][target]]

    embeddings: Dict[int, np.ndarray] = {}

    for target in stored["embeddings"].keys():
        embeddings[int(target)] = np.asarray(stored["embeddings"][target])

    return memory, embeddings

def plot_mean_embeddings(embeddings: Dict[int, np.ndarray]) -> None:
    pca = PCA()
    plt.figure(figsize=(8,6))
    means = np.asarray(list(embeddings.values())[:10])
    Xt = pca.fit_transform(means)
    plot = plt.scatter(Xt[:,0], Xt[:,1], c=list(embeddings.keys())[:10], s=[10]*10)
    labels = list([str(x) for x in embeddings.keys()])
    for i in range(10):
        plt.text(Xt[i,0], Xt[i,1], s=labels[i])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components")
    plt.show()

def plot_all_embeddings(means: Dict[int, np.ndarray], samples: Dict[int, List[np.ndarray]], plot_first_n: int = -1) -> None:
    if plot_first_n == -1:
        plot_first_n = 100

    pca = PCA()
    mean_np = np.asarray([means[x] for x in means.keys()])
    sample_vits = {
        target: convert_samples_to_vit_features(samples[target])
        for target in [23, 8, 19, 6, 84, 70, 58, 92, 32, 79][:plot_first_n]
    }

    sample_np = np.asarray([sample_vits[x] for x in sample_vits.keys()]).reshape(-1, 768)
    all_np = np.concatenate([mean_np, sample_np])
    pca.fit(all_np)
    Xt = pca.transform(all_np)
    # colours = cm.get_cmap("rainbow")(np.linspace(0, 1, plot_first_n))
    plot_first_n = 10

    colours = {
        23: "blue",
        8: "red",
        19: "green",
        6: "yellow",
        84: "black",
        70: "purple",
        58: "aqua",
        92: "gray",
        32: "#00ff11",
        79: "#11ffaa"
    }

    cs = [colours[x] for x in [23, 8, 19, 6, 84, 70, 58, 92, 32, 79]]

    plot = plt.scatter(Xt[:plot_first_n, 0], Xt[:plot_first_n, 1], c=cs, s=[40]*plot_first_n)
    labels = list([x for x in sample_vits.keys()])
    real_labs = ["cloud", "bicycle", "cattle", "bee", "table", "rose", "pickup_truck", "tulip", "flatfish", "spider"]
    for i in range(len(real_labs)):
        plt.text(Xt[i,0], Xt[i,1], s=str(real_labs[i]))
    offset = 0

    for target in [23, 8, 19, 6, 84, 70, 58, 92, 32, 79]:#[:plot_first_n]:
        colour = colours[target]
        print(len(samples[target]), len(Xt[(plot_first_n + offset):(plot_first_n + offset + len(samples[target])), 0]), len(Xt[(plot_first_n + offset):(plot_first_n + offset + len(samples[target])), 1]))
        plt.scatter(
            Xt[(plot_first_n + offset):(plot_first_n + offset + len(samples[target])), 0], 
            Xt[(plot_first_n + offset):(plot_first_n + offset + len(samples[target])), 1], 
            c=[colour for _  in range(len(samples[target]))], 
            s=[4]*len(samples[target])
        )

        offset += len(samples[target])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components")
    plt.show()

@torch.no_grad()
def load_and_plot_embeddings():
    memory, embeddings = load_samples_and_means("ne5_features.json")
    ids = [23, 8, 19, 6, 84, 70, 58, 92, 32, 79]

    memory_two = {}
    embeddings_two = {}

    for id in ids:
        memory_two[id] = memory[id]
        embeddings_two[id] = embeddings[id]

    # plot_mean_embeddings(embeddings)
    plot_all_embeddings(embeddings_two, memory_two, plot_first_n=-1)

def plot_line_graph():
    bounding_lines = {
        # "Offline": 0.6607,
        # "GDumb": 0.3927,
        # "ViT w/ Transfer": 0.9167 # Appendix C of ViT paper
    }
    
    series = {
        "ViT NCM w/ Random": [[1, 2, 3, 4, 5], [0.1734, 0.3207, 0.4555, 0.5874, 0.7031]],
        "ViT NCM w/ Uncertainty": [[1, 2, 3, 4, 5], [0.1720, 0.3250, 0.4693, 0.6076, 0.7317]],
        # "ViT MLP w/ Uncertainty": [[1, 2, 3, 4, 5], [0.1745, 0.3289, 0.4700, 0.6081, 0.7336]],
        # "L2P": [[1, 2, 3, 4, 5], [0.1801, 0.3187, 0.4340, 0.5309, 0.6158]],
        # "DER": [[1, 2, 3, 4, 5], [0.0410, 0.0485, 0.0558, 0.0766, 0.0629]],
        # "DER++": [[1, 2, 3, 4, 5], [0.0507, 0.0669, 0.0759, 0.0962, 0.1014]]
    }

    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 5])

    # Plot techniques with tasks
    for technique in series.keys():
        xs, ys = series[technique]
        plt.plot(xs, ys, label=technique, marker="o")

    # Plot constant lines
    x_space = np.arange(1, 6)
    for technique in bounding_lines.keys():
        val = bounding_lines[technique]
        plt.plot(x_space, np.array(val).repeat(x_space.shape[0]), label=technique, linestyle="dashed")

    plt.yticks([x / 10 for x in range(11)])
    plt.xticks(range(1, 6))
    plt.grid()
    plt.legend(loc="upper left", framealpha=1)
    plt.xlabel("Task")
    plt.ylabel("Overall Accuracy")
    plt.title("CIFAR-100 Task Accuracy")
    plt.show()

plot_line_graph()
# load_and_plot_embeddings()