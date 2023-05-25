from simpletransformers.classification import ClassificationModel
import torch
import streamlit as st
def main():
    chem_class_model = torch.load('chemberta_classification.pth')
    chem_reg_model = torch.load('chemberta_regression.pth')
    
    st.write('# Fine-Tuning Transformers Model for Prediction Bioactivity of Alzheimer’s Drug Candidates')
    st.write('The lack of effectiveness of the wet lab approach to test all potential compounds as drug candidates, incur many in silico (computer-driven) approach has been conducted to reach the efficiency. Prediction of Molecular properties is one of a crucial task in drug discovery. Bioactivity values (e.g., IC50 values) are molecular properties which have commonly used to evaluate and select potential drug candidates. To increase the efficiency in the process of drug discovery, lots of research have been conducted by implementing machine learning to do drug candidate screening. Where, the implementation machine learning such as Random Forest and Extreme Gradient Boosting (XGBoost) can significantly reduce the number of in vitro test. Language model especially Transformers, recently became alternative to solve the challenge in drug discovery which also have been effectively used to generate molecular representation. This research investigate how well ChemBERTa as one of the Transformers models generate canonical SMILES representation and perform prediction on Alzheimer’s Drug Candidates dataset.')
    st.write('')
    st.write('## Fine-Tuning with ChemBERTa')
    input_compound = st.text_input('Input molecule to predict:',placeholder='example input O=C(N1CCCCC1)n1nc(-c2ccc(Cl)cc2)nc1SCC(F)(F)F')
    chem_class_result = chem_class_model.predict([input_compound])
    chem_reg_result = chem_reg_model.predict([input_compound])

    chem_bioactivity_code = chem_class_result[0][0]
    chem_reg_score = str(chem_reg_result[0])

    label_mapping = {1 : "active", 0: "inactive"}
    chem_bioactivity_class = label_mapping[chem_bioactivity_code]

    col1, col2 = st.columns(2)
    if input_compound:
        with col1:
            st.write('### Bioactivity Class')
            st.write(chem_bioactivity_class)

        with col2:
            st.write('### pIC50 Value')
            st.write(chem_reg_score, 'Molar')

    st.write('## Note')
    st.write('compound with IC50 value > 6 Molar indicated as an “active” compound and IC50 value < 5 Molar indicated as an “inactive” compound')
if __name__ == '__main__':
    main()