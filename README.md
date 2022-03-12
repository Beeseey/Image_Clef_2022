# Image_Clef_2022

---->>> Run in Virtual Env (imageclef_env) <<<-------


#########################################################
#########################################################


C O D E      D O C U M E N T A T I O N



####execute with python run.py 

-- run args
[-mode] : [train,test,help]


####for train
[--traincsv] path to the training csv file with row 1 as header and subsequent rows having the format (image_id  concepts)

[--imagepath] path to the images

[--batchsize] batch size, default = 1

[--epochs] epochs, default = 10

[--model] name of model from model.py

[--modeloutput] name of model output

[--valsplit] validation split percentage, default = 0.25

[--classweight_hbar] upper limit for class weights

[--classweight_lbar] lower limit for class weights

[--datatype] type of data passed into model

[--model_path] path for model testing



---->> Example <<----

FOR CONCEPT CLASSIFICATION

python run.py -mode train --traincsv ../../ImageCLEF2022/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv --imagepath ../../ImageCLEF2022/e229cc37-d0da-4356-bd5c-f119c63dfacc_ImageCLEFmedCaption_2022_valid_images/valid --valsplit 0.20 --model concept_classifier --batchsize 5 --epochs 1 --data_type concepts

FOR CONCEPT TRANSFORMER MODEL

python run.py -mode train --traincsv ../../ImageCLEF2022/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv --imagepath ../../ImageCLEF2022/e229cc37-d0da-4356-bd5c-f119c63dfacc_ImageCLEFmedCaption_2022_valid_images/valid --valsplit 0.20 --model transformer --batchsize 5 --epochs 1 --data_type concepts_seq

FOR CAPTION GENERATION

python run.py -mode train --traincsv ../../ImageCLEF2022/cc3d9c72-6c2b-4bd3-9d10-4e133031be48_ImageCLEFmedCaption_2022_caption_prediction_valid.csv --imagepath ../../ImageCLEF2022/e229cc37-d0da-4356-bd5c-f119c63dfacc_ImageCLEFmedCaption_2022_valid_images/valid --valsplit 0.20 --model transformer --batchsize 1 --epochs 1 --data_type captions


Model results and configurations are saved in Results.txt


#########################################################
#########################################################