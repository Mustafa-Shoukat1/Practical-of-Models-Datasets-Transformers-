# Hugging Face Guide: 📚 Usage of Datasets, Pre-trained Models, and Transformers Libraries 
![image](https://github.com/Mustafa-Shoukat1/Practical-of-Models-Datasets-Transformers-/assets/162743520/ceeade02-47f8-4801-8a6e-9efce169e537)


<div style="border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; background-color: #000000; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <h1 style="color: #1976D2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">
        Asalam alaikum warahmatullah wabarakatu!
    </h1>
    <p style="color: #1976D2; font-size: 18px; margin: 10px 0;">
        I am Mustafa Shoukat, deeply engaged in the field of AI. Join me as we explore cutting-edge concepts and techniques, advancing our skills together. In this notebook my aims to examine various Hugging Face models, with the objective of increasing my confidence in applying and customizing pretrained models for different real world tasks.
    </p>
    <p style="color: #43A047; font-size: 16px; font-style: italic; margin: 10px 0;">
        "The power of community lies in our combined efforts. United, we can reach heights we never imagined."
    </p>
    <h2 style="color: #1976D2; margin-top: 15px; font-size: 28px;">Contact Information</h2>
    <table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
        <tr>
            <th style="color: #1976D2; font-size: 18px; padding: 8px; border-bottom: 2px solid #64B5F6;">Name</th>
            <th style="color: #1976D2; font-size: 18px; padding: 8px; border-bottom: 2px solid #64B5F6;">Email</th>
            <th style="color: #1976D2; font-size: 18px; padding: 8px; border-bottom: 2px solid #64B5F6;">LinkedIn</th>
            <th style="color: #1976D2; font-size: 18px; padding: 8px; border-bottom: 2px solid #64B5F6;">GitHub</th>
            <th style="color: #1976D2; font-size: 18px; padding: 8px; border-bottom: 2px solid #64B5F6;">Kaggle</th>
        </tr>
        <tr>
            <td style="font-size: 16px; padding: 8px;">Mustafa Shoukat</td>
            <td style="font-size: 16px; padding: 8px;">mustafashoukat.ai@gmail.com</td>
            <td style="font-size: 16px; padding: 8px;">
                <a href="https://www.linkedin.com/in/mustafashoukat/" target="_blank">
                    <img src="https://img.shields.io/badge/LinkedIn-0e76a8.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge" style="border-radius: 5px;">
                </a>
            </td>
            <td style="font-size: 16px; padding: 8px;">
                <a href="https://github.com/Mustafa-Shoukat1" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-171515.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge" style="border-radius: 5px;">
                </a>
            </td>
            <td style="font-size: 16px; padding: 8px;">
                <a href="https://www.kaggle.com/mustafashoukat" target="_blank">
                    <img src="https://img.shields.io/badge/Kaggle-20beff.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge" style="border-radius: 5px; margin: 0 5px;">
                </a>
            </td>
        </tr>
    </table>
</div>




<div style="border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; background-color: #000000; color: #FFFFFF; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <h1 style="color: #1976D2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">
        Purpose:
    </h1>
    <p style="font-size: 18px; margin: 10px 0;">
        Introduction to Hugging Face Transformers, Datasets and Models
    </p>
    <p style="font-size: 18px; margin: 10px 0;">
        This notebook provides a practical exploration of Hugging Face's ecosystem, focusing on datasets, models, and transformers. Hugging Face has become a pivotal platform for Advance Data Science, offering powerful tools and pre-trained models that streamline the development of various applications.
    </p>
    <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 24px;">
        Transformers and Pipelines
    </h2>
    <p style="font-size: 18px; margin: 10px 0;">
        Hugging Face Transformers enable a wide range of NLP tasks through easy-to-use pipelines. Key functionalities include:
    </p>
    <ul style="font-size: 18px; margin: 10px 0; text-align: left;">
        <li><strong>Named Entity Recognition (NER):</strong> Identifying entities like names, dates, and locations in text.</li>
        <li><strong>Question Answering:</strong> Providing answers to questions posed in natural language.</li>
        <li><strong>Summarization:</strong> Generating concise summaries of input text.</li>
        <li><strong>Translation:</strong> Translating text between different languages.</li>
        <li><strong>Text Classification:</strong> Categorizing text into predefined classes.</li>
    </ul>
    <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 24px;">
        Datasets and Tokenization
    </h2>
    <p style="font-size: 18px; margin: 10px 0;">
        The platform includes access to various datasets and efficient tokenization mechanisms, essential for preparing data for model training and evaluation.
    </p>
    <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 24px;">
        Hugging Face Models
    </h2>
    <p style="font-size: 18px; margin: 10px 0;">
        Hugging Face hosts a repository of state-of-the-art models, including:
    </p>
    <ul style="font-size: 18px; margin: 10px 0; text-align: left;">
        <li><strong>Mixtral-8x22B-v0.1:</strong> A Mixture of Experts model known for its high performance.</li>
    </ul>
    <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 24px;">
        Mixture of Experts Model
    </h2>
    <p style="font-size: 18px; margin: 10px 0;">
        Exploring advanced models like the Mixtral-8x22B-v0.1, demonstrating the capabilities of complex architectures in NLP tasks.
    </p>
    <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 24px;">
        Evaluation Metrics
    </h2>
    <p style="font-size: 18px; margin: 10px 0;">
        Evaluation of models includes metrics such as:
    </p>
    <ul style="font-size: 18px; margin: 10px 0; text-align: left;">
        <li><strong>Bert Score:</strong> Evaluating the quality of text generation.</li>
        <li><strong>BLEU (Bilingual Evaluation Understudy):</strong> Assessing machine-translated text.</li>
        <li><strong>GLUE (General Language Understanding Evaluation):</strong> A benchmark for evaluating language understanding models.</li>
        <li><strong>Perplexity:</strong> Measuring how well a probability model predicts a sample.</li>
    </ul>
    <p style="font-size: 18px; margin: 10px 0;">
        This notebook aims to provide hands-on experience with these tools and concepts, empowering developers to leverage Hugging Face for advanced NLP applications.
    </p>
</div>



# Natural Language Processing with Transformers, Datasets, and SentencePiece

## Overview

This notebook demonstrates the use of state-of-the-art Natural Language Processing (NLP) tools and libraries to build and fine-tune models for various NLP tasks. We will leverage Hugging Face's `Transformers` and `Datasets` libraries alongside Google's `SentencePiece` for tokenization and text normalization.

## Libraries and Tools

### Transformers

Developed by [Hugging Face](https://huggingface.co/), the `Transformers` library provides a variety of pre-trained models that have revolutionized NLP. These models, such as BERT, GPT, and RoBERTa, are designed for tasks like:

- Text classification
- Question answering
- Text generation
- Language translation

Key features:
- Easy loading and fine-tuning of pre-trained models.
- Interfaces for both TensorFlow and PyTorch.

### Datasets

Also developed by [Hugging Face](https://huggingface.co/), the `Datasets` library offers high-quality datasets for training and evaluating NLP models. It supports tasks including:

- Text classification
- Named entity recognition (NER)
- Machine translation

Key features:
- Unified interface for accessing and manipulating datasets.
- Functions for downloading, splitting, preprocessing, and loading datasets.

### SentencePiece

Developed by Google, `SentencePiece` is a tokenizer and text normalizer that segments text into subword units. This approach is particularly useful for handling out-of-vocabulary words and reducing vocabulary size.

Key features:
- Uses a unigram language model to learn tokenization from text.
- Breaks new text into subwords based on learned patterns.
- Compatible with NLP frameworks like TensorFlow and PyTorch.

## Workflow

1. **Loading Pre-trained Models and Datasets:**
   - Utilize the `Transformers` library to load state-of-the-art pre-trained models.
   - Access and manipulate datasets using the `Datasets` library.

2. **Tokenization and Text Normalization:**
   - Apply `SentencePiece` for effective tokenization, which helps in managing out-of-vocabulary words and reduces the vocabulary size.

3. **Fine-tuning and Evaluation:**
   - Fine-tune the pre-trained models on specific NLP tasks.
   - Evaluate model performance using the provided datasets.

## Conclusion

By integrating `Transformers`, `Datasets`, and `SentencePiece`, we can efficiently build and fine-tune robust NLP models capable of handling a variety of tasks. This notebook serves as a comprehensive guide to harnessing the power of these advanced tools in your NLP projects.

# **Transformers Pipeline**

The core architecture behind the text classification pipeline in the Hugging Face Transformers library typically involves a pre-trained transformer model fine-tuned on a large corpus of text data for the specific task of text classification. Here's how it works, illustrated with an example:

## **Loading the Model:**

When you create a text classification pipeline using `pipeline('text-classification')`, the Hugging Face library automatically downloads the pre-trained model associated with text classification tasks. This model is typically a transformer-based architecture like BERT, RoBERTa, or DistilBERT that has been fine-tuned on a dataset for text classification.

## **Tokenization:**

The input text provided to the pipeline is tokenized using the same tokenizer that was used during the pre-training of the model. This tokenizer converts the input text into a sequence of tokens that can be processed by the transformer model.

## **Encoding:**

The tokenized input text is then encoded into numerical representations suitable for input to the transformer model. This encoding typically involves mapping each token to its corresponding index in the model's vocabulary and converting the input sequence into numerical vectors.

## **Model Inference:**

The encoded input sequence is fed into the pre-trained transformer model. The model processes the input sequence through multiple layers of self-attention mechanisms and feed-forward neural networks to extract contextual representations of the text.

## **Classification Head:**

After processing the input text through the transformer layers, a classification head is attached to the model. This classification head consists of one or more dense layers that take the final contextual representation of the input text as input and produce predictions for the classification task.

- For example, in binary classification tasks, the classification head may consist of a single neuron with a sigmoid activation function that outputs a probability score indicating the likelihood of the input text belonging to a particular class.
- In multi-class classification tasks, the classification head may consist of multiple neurons corresponding to each class, with a softmax activation function applied to produce a probability distribution over the classes.

## **Prediction:**


Finally, the model outputs the predicted class label or probability distribution over the classes for the input text. This prediction is returned as the output of the text classification pipeline.

<div style="border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; background-color: #1B5E20; color: white; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">
        I appreciate your help with this notebook
    </h1>
    <h2 style="color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); font-weight: normal; margin-bottom: 10px; font-size: 24px;">
       📓 Your feedback will be important as I try more advanced techniques.🚀
    </h2>
    <h2 style="color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); font-weight: normal; margin-bottom: 10px; font-size: 24px;">
        Suggestions are always welcome!
    </h2>
    <h2 style="color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); font-weight: normal; font-size: 24px;">
        Let's keep pushing forward together!
    </h2>
</div>


# Next Explore Notebook for Practical Guide of Hugging Face 📖: Models 🧩, Transformers 🤖, and Pipelines 🔄🤗

