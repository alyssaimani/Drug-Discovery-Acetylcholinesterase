# Drug_Discovery_Acetylcholinesterase
The work of this project aims to study the application of machine learning in the field of bioinformatics, especially Drug Discovery.
This project was created by following a tutorial from [Chanin Nantasenamat](https://github.com/dataprofessor). 
Then this project continued for the Natural Language Processing coursework at Binus University by fine-tuning the dataset with Transformers model ([ChemBERTa](https://github.com/deepchem/deepchem)).

# Background
The lack of effectiveness of the wet lab approach to test all potential compounds as drug candidates, 
incur many in silico (computer-driven) approach has been conducted to reach the efficiency.
Prediction of Molecular properties is one of a crucial task in drug discovery. 
Bioactivity values (e.g., IC50 values) are molecular properties which have commonly used to evaluate and select potential drug candidates. 
To increase the efficiency in the process of drug discovery, lots of research have been conducted by implementing machine learning to do drug candidate screening. 
Where, the implementation machine learning such as Random Forest and Extreme Gradient Boosting (XGBoost) can significantly reduce the number of in vitro test. 
Language model especially Transformers, recently became alternative to solve the challenge in drug discovery 
which also have been effectively used to generate molecular representation. 
This research investigate how well ChemBERTa as one of the Transformers models generate canonical SMILES representation and 
perform prediction on Alzheimer’s Drug Candidates dataset. 

# How to Run the Streamlit App
 ```shell
streamlit run app.py
```
# Requirement
```shell
transformers==4.24.0
streamlit==1.22.0
torch==2.0.1
torchaudio==2.0.2
torchvision==0.15.2
simpletransformers==0.63.11
```
# References
https://github.com/dataprofessor/bioinformatics_freecodecamp

