"""
What this does:
- generates patent titles from abstracts using the latest checkpoint
- uses HuggingFace's GPT2 model.generate() method

If you want to run this with your own abstracts
(such as those found here https://www.uspto.gov/patents/search/patent-public-search)
then what you should do is paste them in the "abstracts" list at the end of the config
"""

"""
before running this, run:

!pip install transformers tokenizers

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
"""

"""
See LLM note at the top of tokenize_data.py
"""

import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer


#paths
prepared_data_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/prepared_data"
checkpoint_to_load = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/checkpoints/ckpt_epoch_008.pt"

#must match prepare_data.py and train.py
block_size = 512

#special token strings (must match prepare_data.py)
pad_token = "<pad>"
bos_token = "<bos>"
sep_token = "<sep>"
eos_token = "<eos>"

#sampling
max_new_tokens = 64
temperature = 0.8
top_k = 40
number_of_samples = 3

#abstracts to generate titles for
abstracts = [
    "Disclosed are systems and methods for predicting patient response to a treatment option. In one embodiment, the image slides from patient tissue samples are divided into patches and morphological patterns correlated with a disease outcome are labeled and given a patch-level score, based on whether the morphological patterns occur only in patients with good outcomes or patients with poor outcomes. A patient-level score can be generated based, at least partly, on the patch-level scores. Patch-level scores can identify regions of interest for targeted biomarker identification.",
    "Provided herein are methods of determining the size and purify of nucleic acids (e.g., mRNAs) by using hydrophilic interaction chromatography (HILIC)-based methods to separate the nucleic acids from a mixture, followed by mass spectrometry to determine the size of the nucleic acids.",
    "Provided is a method to remove double-stranded DNA with sequence errors, which can occur in various stages of DNA production such as chemical synthesis, hybridization, and amplification, from double-stranded DNA without sequence errors, thereby providing double-stranded DNA with a low proportion of sequence errors. Specifically, this invention is a method for producing double-stranded DNA, which includes: (1) providing a double-stranded DNA mixture containing double-stranded DNA with sequence errors and double-stranded DNA without sequence errors; (2) adding a mismatch repair-related enzyme group to the double-stranded DNA mixture, where the mismatch repair-related enzyme group includes MutS and MutL, or MutS and single-strand specific exonuclease; and (3) subjecting the double-stranded DNA mixture to a double-stranded DNA amplification reaction.",
    "Provided is a method capable of simply and exponentially amplifying circular DNA, and particularly, long-chain circular DNA, in a cell-free system. Specifically, provided herein is a method for amplifying circular DNA which comprises mixing circular DNA having a replication origin sequence (origin of chromosome (oriC)) with a reaction solution comprising: a first enzyme group that catalyzes replication of circular DNA; a second enzyme group that catalyzes an Okazaki fragment maturation and synthesizes two sister circular DNAs constituting a catenane; a third enzyme group that catalyzes a separation of two sister circular DNAs; and also, a buffer, NTP, dNTP, a magnesium ion source, and an alkali metal ion source, to form a reaction mixture, which is then reacted.",
    "An intuitive interface may allow users of a computing device (e.g., children, etc.) to create imaginary three dimensional (3D) objects of any shape using body gestures performed by the users as a primary or only input. A user may make motions while in front of an imaging device that senses movement of the user. The interface may allow first-person and/or third person interaction during creation of objects, which may map a body of a user to a body of an object presented by a display. In an example process, the user may start by scanning an arbitrary body gesture into an initial shape of an object. Next, the user may perform various gestures using his body, which may result in various edits to the object. After the object is completed, the object may be animated, possibly based on movements of the user.",
    "Embodiments of the present disclosure include systems and methods for sparsifying narrow data formats for neural networks. A plurality of activation values in a neural network are provided to a muxing unit. A set of sparsification operations are performed on a plurality of weight values to generate a subset of the plurality of weight values and mask values associated with the plurality of weight values. The subset of the plurality of weight values are provided to a matrix multiplication unit. The muxing unit generates a subset of the plurality of activation values based on the mask values and provides the subset of the plurality of activation values to the matrix multiplication unit. The matrix multiplication unit performs a set of matrix multiplication operations on the subset of the plurality of weight values and the subset of the plurality of activation values to generate a set of outputs.",
    "A system and method and for method for optimizing performance of a natural language processing (NLP) model includes clustering a validation dataset used in training the NLP model into a plurality of clusters; measuring a generalization in context parameter for one or more of the plurality of clusters; measuring an interference in context parameter for one or more of the plurality of clusters; and identifying a cluster, from among the plurality of clusters, for data augmentation, based on the measured generalization in context parameter and the measured interference in context parameter. Once a cluster is identified, a prompt is generated for submission as an input to a large language model (LLM) to prompt the LLM to automatically generate synthetic training data for the identified cluster, before the prompt is provided to the LLM and synthetic training data is received from the LLM. The synthetic training data is then labeled by a human before being used to further train the NLP model to improve the performance of the NLP model with respect to the identified cluster.",
    "Methods and computing devices for estimating a force F exerted on a touchpad are disclosed. In one example, a method comprises determining that the touchpad is not being touched. At least on condition of determining that the touchpad is not being touched, a no-touch capacitance value of the PCB is calculated. After calculating the no-touch capacitance value, the method includes determining that the touchpad is being touched. At least on condition that the touchpad is being touched, the no-touch capacitance value and a touch-based capacitance value are used to estimate the force F exerted on the touchpad."
]

def load_model_from_checkpoint(checkpoint_path):
    print(f"Loading {os.path.basename(checkpoint_path)}")
    state = torch.load(checkpoint_path, map_location="cuda")
    config = GPT2Config(**state["config"])
    model = GPT2LMHeadModel(config).to("cuda")
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Model loaded. Epoch={state['epoch']}, step={state['step']}")
    return model

@torch.no_grad()
def generate_title(model, tokenizer, abstract):
    bos_id = tokenizer.token_to_id(bos_token)
    sep_id = tokenizer.token_to_id(sep_token)
    eos_id = tokenizer.token_to_id(eos_token)
    pad_id = tokenizer.token_to_id(pad_token)
    abstract_ids = tokenizer.encode(abstract).ids

    #truncating the abstract if its too long for the context length
    max_abstract_length = block_size - 2 - max_new_tokens
    if len(abstract_ids) > max_abstract_length:
        abstract_ids = abstract_ids[-max_abstract_length:]

    prefix = [bos_id] + abstract_ids + [sep_id] #this is what the model starts from when generating a title
    input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")

    #generating the output
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=top_k, temperature=temperature, eos_token_id=eos_id, pad_token_id=pad_id)

    #converting the output to something human readable
    generated_ids = output[0, len(prefix):].tolist()
    if generated_ids and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    output = tokenizer.decode(generated_ids).strip()
    return output

def main():
    #loading stuff
    model = load_model_from_checkpoint(checkpoint_to_load)
    tokenizer = ByteLevelBPETokenizer(vocab=f"{prepared_data_directory}/vocab.json", merges=f"{prepared_data_directory}/merges.txt")

    #generating titles for each abstract
    for i, abstract in enumerate(abstracts, 1):
        print()
        print(f"Abstract {i}")
        print(abstract)
        print()
        for sample in range(1, number_of_samples + 1):
            title = generate_title(model, tokenizer, abstract)
            print(f"[{sample}] {title}")
        print()

if __name__ == "__main__":
    main()