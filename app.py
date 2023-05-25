from simpletransformers.classification import ClassificationModel
import torch
import streamlit as st
def main():
    chem_class_model = torch.load('chemberta_classification.pth')
    chem_reg_model = torch.load('chemberta_regression.pth')
    
    st.write('# Prediction Bioactivity of Alzheimer’s Drug Candidates using NLP based Approach')
    st.write('The Process of drug discovery can take decades and cost billions of dollars until it is approved by FDA. Prediction of Molecular properties is one of a crucial task in drug discovery. Bioactivity values (e.g., IC50 values) are molecular properties which have commonly used to evaluate and select potential drug candidates. To increase the efficiency in the process of drug discovery, lots of research have been conducted by implementing machine learning to do drug candidate screening. In this study, we conduct NLP based approach by fine-tuning of Transformers Model on Alzheimer’s drug candidates to predict their bioactivity. We conduct fine-tuning on classification and regression tasks. For comparison we compare the result with QSAR random forest model and logistic regression.')
    st.write('')
    st.write('## Fine-Tuning with ChemBERTa')
    input_molecule = st.text_input('Input molecule to predict:',placeholder='example input CN(C(=O)n1nc(-c2ccc(Cl)cc2)nc1SCC(F)(F)F)c1ccccc1')
    chem_class_result = chem_class_model.predict([input_molecule])
    chem_reg_result = chem_reg_model.predict([input_molecule])

    chem_bioactivity_code = chem_class_result[0][0]
    chem_reg_score = str(chem_reg_result[0])

    label_mapping = {1 : "active", 0: "inactive"}
    chem_bioactivity_class = label_mapping[chem_bioactivity_code]

    col1, col2 = st.columns(2)
    if input_molecule:
        with col1:
            st.write('### Classification Result')
            st.write(chem_bioactivity_class)

        with col2:
            st.write('### Regression Result')
            st.write(chem_reg_score)

if __name__ == '__main__':
    main()