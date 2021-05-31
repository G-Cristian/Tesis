#!/bin/bash

prefix_name="Surface-FPS"
date_time=$(date +"%Y_%m_%d_%H_%M_%S")

name=$prefix_name'_'$date_time
outputDir="../results/$name"

logfile="../results/$name/log.txt"
train_args="--outputDir $outputDir --samplingMethod $prefix_name --epochLengthPow 6 --useNormals 0"

echo "Inicio -- $date_time --"

echo "Creando carpeta $outputDir"
mkdir "$outputDir"

echo "Creando archivo de log $logfile"
touch "$logfile"

echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "Inicio -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

obj_file="armadillo.obj"

echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
echo "Ejecutando trainer.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"

echo "python3 trainer.py ../data/$obj_file $train_args"
echo "python3 trainer.py ../data/$obj_file $train_args" >> "$logfile"

python3 trainer.py ../data/$obj_file $train_args

echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "... Fin ejecutando trainer.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..."
echo "Ejecutando modelmesher.py para $obj_file ($(date +"%Y_%m_%d_%H_%M_%S")) ..." >> "$logfile"

echo "python3 modelmesher.py $outputDir"
echo "python3 modelmesher.py $outputDir" >> "$logfile"

python3 modelmesher.py $outputDir

echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "... Fin ejecutando modelmesher.py para $obj_file -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --"
echo "Fin -- $(date +"%Y_%m_%d_%H_%M_%S") --" >> "$logfile"

exit

