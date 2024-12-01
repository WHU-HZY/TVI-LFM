# Generators

### Preparing

Install the library from Huggingface:
```shell
pip install transformers==4.32.1

pip install fschat==0.2.36
```

Then download the well-trained [IR-CAPTION.tar.gz](https://drive.google.com/file/d/17nOBeGHf4r4MHSeuFut4Pf-4m8ZSPPjC/view?usp=drive_link) and [RGB-CAPTION.tar.gz](https://drive.google.com/file/d/1_w751YFyBLnnVBcnnLK4gFRvbrng1t1E/view?usp=drive_link), put them in the `generators/weights/IR_CAPTION` directory and `generators/weights/RGB_CAPTION` directory, respectively. Then decompress them.

Now we have the file tree like:

```
weights
├── IR_CAPTION
│   ├── checkpointxxxx/
│   ├── IR-CAPTION.tar.gz
├── RGB_CAPTION
│   ├── checkpointxxxx/
│   ├── RGB-CAPTION.tar.gz
```

### Guidance


`demo/generator_demo.ipynb`: The textual expanding code for IR/RGB person images. (We can get text descriptions for infrared images without color and those for visible images with color.)

`demo/llm_rephrase.py`: The code for rephrasing generated texts with LLM.
```
python generators/code/llm_rephrase.py --input 'Her black shoes are a matching sky blue red' --gpus 0
```

`demo/color_mover.ipynb` is a tool aims to remove the color in the rephrased infrared image texts caused by LLM hallucination. (**Note**: the BLIP generated infrared image texts do not contain any color, to mitigate this hallucination we can manually add prompts against color representation while LLM rephrasing infrared image texts.)


