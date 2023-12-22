# Workplan & Logbook of activities 

--------------------------------------------------------------

**Group members**:
Group FLOOD2
- Antonio Magherini - 5838215
- David Najda - 
- Apostolos Politis - 5761387
- Carlo Sobral de Vito - 
- Lucas Terlinden-Ruhl - 5863937

**Project title**:
Spatio-temporal flood modelling with Deep Learning

**Background**:
%%%% to be added

**Objectives**:
%%%% to be added

**Project planning**:
Finish first minimum model by week 2.7 before the Mid-term review.
In the meantime implement at least another model and prepare PowerPoint slides for Mid-term review.

Future works:
- implement other models (by around 18/01);
- apply data augmentation (by around 18/01);
- generalize models to ddatasets 2 & 3 (by aroud 28/01);
- through Delft3D create new datasets (by aroud 28/01).

If time allows it, other models from literature will be explored. 

**Team organisation**:
Apostolos, David, Carlo: input data preparation, CNN model, other to be discussed.
Lucas, Antonio: ConvLSTM model, other to be discussed.

Everyone: PP presentation, report, other to be discussed.

### Logbook of activities

15/12/2023 - Antonio, David, Lucas present (1.5 h). Apostolos joined later (0.5 h). Carlo absent. 
- Project introduction by Roberto, initial meeting and first share of ideas.

16/12/2023 – Antonio (0.5 h). 
- Creation of GitHub project and repository, Logbook of activities and Workplan template.

17/12/2023 – Apostolos (2.5 h). Studying project's given dataset and python code. Realization of given dataset structure.

19/12/2023 – Apostolos and David meeting (0.5 h). Discussion about the given dataset structure. Discussion about Interaction Visualization and it's possible application on the project.
- Apostolos (0.5h). Discussion with project's tutor about additional instructions, explanations and possible progress steps.
- Apostolos (4h). Initial edit of given data for easier implementation on workshop's given CNN python code.

20/12/2023 – Everyone present (1.5 h). 
- Meeting online. Discussion on work done so far by Apostolos and David (CNN setup), expected outputs and model architecture. 
- Tasks division: Apostolos, David and Carlo working on CNN (+ data augmentation if time allows it), Lucas and Antonio working on ConvLSTM once data processing is done from CNN model.
- Apostolos and David (0.5h). Discussion with project's tutor about further explanation of given dataset. Updates on current progress. Discussion about implementation of GNN models on project.

21/12/2023 - Antonio (1 h). GitHub repository improvement (folders and files creation). Added datasets and other files from Google Drive.
- Apostolos (1h). Further edit on given data for easier implementation on workshop's given CNN python code.
- Apostolos and David. Presentation of progress to the project's tutor and discussion about further explanation on workshop's given CNN python code.
- Apostolos, David and Calro (6h). Finalization of given dataset edit. Implementation on workshop's given CNN python code. First draft CNN model (single step prediction).

22/12/2023 - Anotonio and Lucas (6h)
- Read literature to find a conv LSTM model that used pytorch on github
- Looked through the git folders to see what had been accomplished by other members
- Started implementing cnn model into git, not completed, work can be seen on a seperate branch
- Worked on pre processing all the inputs and targets (including discharge) for all samples and time steps
- Started working on encoding and decoding the csv files to reduce running time of processing the data