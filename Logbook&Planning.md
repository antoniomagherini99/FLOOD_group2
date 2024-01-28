# Workplan & Logbook of activities 

--------------------------------------------------------------

**Group members**:
Group FLOOD2
- Antonio Magherini - 5838215
- David Najda - 4651081
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

16/12/2023 
- Antonio (0.5 h). 
1. Creation of GitHub project and repository, Logbook of activities and Workplan template.

17/12/2023 
- Apostolos (2.5 h). 
1. Studying project's given dataset and python code. 
2. Realization of given dataset structure.
- David (2 h).
1. Studying reference papers + making initial notes for the report.

19/12/2023 
- Apostolos and David meeting (0.5 h). 
1. Discussion about the given dataset structure. Discussion about Interaction Visualization and it's possible application on the project.
- David (2.5 h).
1. Exploration Interactive Visualization (using Vega-Altair) and its applicability on the project.

- Apostolos (0.5h). 
1. Discussion with project's tutor about additional instructions, explanations and possible progress steps.

- Apostolos (4 h). 
1. Initial edit of given data for easier implementation on workshop's given CNN python code.

20/12/2023 – Everyone present (1.5 h). 
1. Meeting online. Discussion on work done so far by Apostolos and David (CNN setup), expected outputs and model architecture. 
2. Tasks division: Apostolos, David and Carlo working on CNN (+ data augmentation if time allows it), Lucas and Antonio working on ConvLSTM once data processing is done from CNN model.
- Apostolos and David (0.5h). 
1. Discussion with project's tutor about further explanation of given dataset. 
2. Updates on current progress. 
3. Discussion about implementation of GNN models on project.

21/12/2023 
- Antonio (1 h). 
1. GitHub repository improvement (folders and files creation). Added datasets and other files from Google Drive.
- Apostolos (1 h). 
1. Further edit on given data for easier implementation on workshop's given CNN python code.
- Apostolos and David. 
1. Presentation of progress to the project's tutor and discussion about further explanation on workshop's given CNN python code.
- Apostolos, David and Carlo (6 h). 
1. Finalization of given dataset edit. 
2. Implementation on workshop's given CNN python code. 
3. First draft CNN model (single step prediction).
- David (1 h).
1. Set up a draft version for the final report + cover page design

22/12/2023 
- Antonio and Lucas (6 h). 
1. Read literature to find a conv LSTM model that used pytorch on github. 
2. Looked through the git folders to see what had been accomplished by other members. 
3. Started implementing CNN model into git, not completed, work can be seen on a seperate branch. 
4. Worked on pre processing all the inputs and targets (including discharge) for all samples and time steps. 
5. Started working on encoding and decoding the csv files to reduce running time of processing the data.

23/12/2023 
- Antonio (4 h). 
1. Input data preprocessing (for training/validation dataset)

24/12/2023
- Lucas and Antonio (3.5 h).
1. Automatization of input data processing for all other datasets
2. Includes git debugging to merge branches

25/12/23
- Lucas (4 h)
1. Worked on post processing the data to ensure data made sense
2. Animation script can be found in conv lstm notebook. Skeleton should be solid. Only need to add more subplots and change the colors.
   To do: Might need to transform into a class to automate calling it.
3. Started working on initializing the conv lstm model

26/12/23
- Lucas (2 h)
1. Worked on train_epoch function for conv lstm model. Not yet functional, but model understands first inputs.
   Having difficulty using the previous outputs as new inputs

30/12/23
- Apostolos (2 h)
1. First attempt on finalizing the first CNN model with 96 outputs instead of 1
- Antonio (1.5 h)
1. Worked on ConvLSTM model.

31/12/23
- Apostolos (3 h)
1. First CNN model with 96 outputs working correctly.

1/1/24
- Apostolos (3 h)
1. Running the first working CNN model multiple times for 48 outputs, figuring out which settings improve the quality of the outcome

8/1/24
- David (1 h)
1. Wrote introduction + notes on dataset sections for the report

9/1/24 - Everyone present (1 h)
1. Progress update
2. Discussed mid-term presentation + task division
- David (2.75 h)
1. Draft design of PowerPoint to be used during mid-term presentation
2. Restructuring of CNN notebook + moving functions to separate .py files. 
- Lucas and Antonio (2.5 h)
1. Updated py files for pre processing so that they return datasets
2. Created a py file for the animations in post processing
3. Tried to debug the conv_lstm model with no success

10/1/24
- Lucas and Antonio (2h)
1. Able to get predictions from conv lstm
2. Realized that loss is extremelly small for bad predictions
- Lucas (2.5h)
0. Met with Roberto to discuss the project
1. Updated cool_animation to include predictions
2. Tried to get better predictions, model performs better with more epochs
   Model also performs better for water depth

11/1/24
- Lucas (4h)
1. Played around with conv lstm model and ran the model for more epochs
2. Attempted to denormalize inputs within cool animation
   Should start to divide that function into multiple ones, making it easier to use
- Lucas and Antionio (2h)
1. Preperation of slides

12/1/24
- Lucas (1h)
1. Split conv_lstm notebooks into two, one of them is now only for training
   The other is only for testing
- All (2h)
1. Prepare for interim presentation and present.
- Antonio and Lucas (2.5h)
1. Implemented feedback to normalize and denormalize the targets seperately
2. Added a 1x1 kernel to the conv lstm model to increase the number of hidden dimensions.
- Antonio (0.5 h)
1. Run new model

13/1/24
- Antonio and Lucas (1h)
1. Reviewed outputs of 'new' conv lstm model, realised it has the same test loss, even though validation loss was decrease by almost 2 orders of magnitude
- Lucas (4h)
1. Worked on post processing to implement feedback. Made it clearer where BC was located and added a new subplot for loss per frame.
2. Code cleanup, to make functions more versatile and easier to change.

17/1/24
- David (1h)
1. Studying CNN model code and making initial notes for the report

18/1/24
- David (4h)
1. Added way to visualize the model architecture + tested this on the CNN model. Tried (unsuccessfully to make the resulting visualization more legible by grouping blocks. 
2. Wrote sub-sections on CNN data pre-processing, model architecture and model training for the report.

23/1/24
- Lucas (2h)
1. Started implementing more metrics to test the ability of each model
- Antonio (3h)
1. Started implementing data augmentation to reduce the loss of dataset 2. Rotation and flipping were considered. The ease of flipping makes it attractive.

24/1/24
- Lucas (5 h)
1. Finished implementing metrics
2. Worked on making convlstm model only work with a single layer instead of 48
- Antonio (6 h) 
1. Worked on data augmentation
2. Started working on post processing plots to show in which situation the model works better
- Lucas And Antonio (2h)
1. Attended feedback session, realised there were mistakes in loading datasets, standardisation and convlstm model implementation

25/1/24
- Lucas (3.5 h)
1. Succesffuling implemented single layer convlstm model
2. Removed pottential conflicts so that all functions could work regardless of model used
- Antonio (8h)
1. Worked on feedback recieved during yesterday's session
