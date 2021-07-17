#!/bin/bash

prefix_name="Surface-FPS"
date_time=$(date +"%Y_%m_%d_%H_%M_%S")

name=$prefix_name'_'$date_time
outputDir="../results/$name"

logfile="../results/$name/log.txt"
completeLogfile="../results/$name/complete_log.txt"
train_args="--outputDir $outputDir --samplingMethod $prefix_name --epochLengthPow 5 --useNormals 1"
#train_args="--outputDir $outputDir --samplingMethod $prefix_name --epochLengthPow 5"
#train_args="--outputDir $outputDir --showVis 1 --reconstructionRes 128"

echo "Inicio -- $date_time --"

echo "Creando carpeta $outputDir"
mkdir "$outputDir"

#echo "Creando carpeta $outputDir/checkpoints"
#mkdir "$outputDir/checkpoints"

echo "Creando archivo de log $logfile"
touch "$logfile"

echo "Creando archivo de log completo $completeLogfile"
touch "$completeLogfile"

echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$completeLogfile"

modelName="armadillo"
#modelName="bumpy-cube"
#modelName="36372"
mesh_ext="obj"
#mesh_ext="stl"
obj_file="$modelName.$mesh_ext"

data_path="../data"
#data_path="../dataset_13_15"

resultsName="results_$modelName"

echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"
echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$completeLogfile"

echo "python3 trainer.py $data_path/$obj_file $train_args"
echo "python3 trainer.py $data_path/$obj_file $train_args" >> "$logfile"
echo "python3 trainer.py $data_path/$obj_file $train_args" >> "$completeLogfile"

python3 trainer.py $data_path/$obj_file $train_args >> "$completeLogfile"

echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$completeLogfile"

echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"
echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$completeLogfile"

echo "python3 modelmesher.py $outputDir"
echo "python3 modelmesher.py $outputDir" >> "$logfile"
echo "python3 modelmesher.py $outputDir" >> "$completeLogfile"

python3 modelmesher.py $outputDir >> "$completeLogfile"

echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$completeLogfile"

echo "Ejecutando metrics.py ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
echo "Ejecutando metrics.py ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"
echo "Ejecutando metrics.py ($(date +"%Y_%m_%d_%H_%M_%S")) ..."  >> "$completeLogfile"

echo "python3 metrics.py $outputDir $data_path/ --meshExt $mesh_ext --resultsName $resultsName"
echo "python3 metrics.py $outputDir $data_path/ --meshExt $mesh_ext --resultsName $resultsName" >> "$logfile"
echo "python3 metrics.py $outputDir $data_path/ --meshExt $mesh_ext --resultsName $resultsName" >> "$completeLogfile"

python3 metrics.py $outputDir $data_path/ --meshExt $mesh_ext --resultsName $resultsName >> "$completeLogfile"

echo "... Fin ejecutando metrics.py -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "... Fin ejecutando metrics.py -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
echo "... Fin ejecutando metrics.py -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$completeLogfile"

echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$completeLogfile"

exit

