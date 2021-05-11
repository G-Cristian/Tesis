#!/bin/bash

prefix_name="Importance"
date_time=$(date +"%Y_%m_%d_%H_%M_%S")

name=$prefix_name'_'$date_time
outputDir="../results/$name"

logfile="../results/$name/log.txt"
train_args="--outputDir $outputDir --samplingMethod Surface"
#train_args="--outputDir $outputDir --epochLengthPow 6 --epochs 1000 --writeOutEpochs 1"
#train_args="--outputDir $outputDir --showVis 1 --reconstructionRes 128"

echo "Inicio -- $date_time --"

echo "Creando carpeta $outputDir"
mkdir "$outputDir"

#echo "Creando carpeta $outputDir/checkpoints"
#mkdir "$outputDir/checkpoints"

echo "Creando archivo de log $logfile"
touch "$logfile"

echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

for obj_file in ../data/*.obj; do
    [ -f "$obj_file" ] || break

    echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
    echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"
    echo "python3 trainer.py ../data/$obj_file $train_args"
    python3 trainer.py ../data/$obj_file $train_args

    echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
    echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
    
    echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
    echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"
    
    python3 modelmesher.py $outputDir
    
    echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
    echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"
done

#echo "Ejecutando metrics.py ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
#echo "Ejecutando metrics.py ($(date +"%Y_%m_%d_%H_%M_%S")) ..."  >> "$logfile"
#python3 metrics.py $outputDir ../data/

#echo "... Fin ejecutando metrics.py -- $(date +"%Y_%m_%d_%H_%M_%S") --"
#echo "... Fin ejecutando metrics.py -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

exit

