# Detectarea Fraudelor folosind Machine Learning

Acest proiect abordează problema detectării fraudelor utilizând tehnici de învățare automată (machine learning). Proiectul include codul pentru preprocesarea datelor, antrenarea modelului și implementarea unui model de predicție a fraudelor.

## Cuprins

- [Introducere](#introducere)
- [Instalare](#instalare)
- [Utilizare](#utilizare)
- [Documentație](#documentație)

## Introducere

În ultimii ani, frauda cu carduri de credit a devenit o preocupare majoră datorită volumului tot mai mare de tranzacții online. Această lucrare explorează aplicarea algoritmilor de învățare automată pentru a îmbunătăți detectarea tranzacțiilor frauduloase în timp real. Am utilizat modele predictive precum Random Forest, XGBoost și rețele neuronale, care au demonstrat o acuratețe ridicată în identificarea tranzacțiilor frauduloase. Aplicația include funcționalități pentru compararea modelelor și vizualizarea tiparelor tranzacțiilor, ajutând utilizatorii să înțeleagă și să atenueze riscurile de fraudă.

## Instalare

Pentru a rula acest proiect, trebuie să aveți instalat R și bibliotecile necesare.

```sh
install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "gbm", "nnet"))
