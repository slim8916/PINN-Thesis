#!/bin/bash

# Output CSV file
output_file="$1.csv"

# Write the header to the CSV file
echo "Trainable params,Non-trainable params,Adam Training Time,Adam Validation Physics Loss,Adam Validation Data Loss,Slope Physics Loss,Slope Data Loss,Slope Total Loss,LBFGSB Optimization Time,LBFGSB Validation Physics Loss,LBFGSB Validation Data Loss" > $output_file



# Iterate over all folders named model_i
for i in $(seq 1 6300); do
    folder="./Res/$1/model_$i"
    txt_file="$folder/result.txt"
    if [ -d "$folder" ]; then
        if [ -f "$txt_file" ]; then
            # Extract values from the txt file
            tr_pars=$(grep 'Trainable params: ' "$txt_file" | awk '{print $3}')
            ntr_pars=$(grep 'Non-trainable params: ' "$txt_file" | awk '{print $3}')
            adam_time=$(grep 'Adam Training Time: ' "$txt_file" | awk '{print $4}')
            adam_phys_ls=$(grep 'Adam Validation Physics Loss: ' "$txt_file" | awk '{print $5}')
            adam_data_ls=$(grep 'Adam Validation Data Loss: ' "$txt_file" | awk '{print $5}')
            slope_phys=$(grep 'Slope Physics Loss: ' "$txt_file" | awk '{print $4}')
            slope_data=$(grep 'Slope Data Loss: ' "$txt_file" | awk '{print $4}')
            slope_total=$(grep 'Slope Total Loss: ' "$txt_file" | awk '{print $4}')
            lbfgsb_time=$(grep 'LBFGSB Optimization Time: ' "$txt_file" | awk '{print $4}')
            lbfgsb_phys_ls=$(grep 'LBFGSB Validation Physics Loss: ' "$txt_file" | awk '{print $5}')
            lbfgsb_data_ls=$(grep 'LBFGSB Validation Data Loss: ' "$txt_file" | awk '{print $5}')

            tr_pars="${tr_pars//,/}"
            ntr_pars="${ntr_pars//,/}"
            # Append values to the CSV file
            echo "$tr_pars,$ntr_pars,$adam_time,$adam_phys_ls,$adam_data_ls,$slope_phys,$slope_data,$slope_total,$lbfgsb_time,$lbfgsb_phys_ls,$lbfgsb_data_ls" >> $output_file
        else
            echo "Warning: No result.txt file found in $folder"
        fi
    else
        echo "Warning: Directory $folder does not exist"
    fi
done

echo "Data extraction completed. Check the $output_file for results."
