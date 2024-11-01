# python npyify.py convert_to_npy --data_root /mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1 \
#         --output_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2
# python npyify.py make_others --data_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2
# python npyify.py consolidate_metadata --data_root /mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1 
# python npyify.py make_split --metadata_path /mnt/gestalt/home/ddmanddman/moisesdb/metadata.csv
# python npyify.py consolidate_stems --data_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2
# python npyify.py get_dbfs_by_chunk --data_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2 \
#                 --query_root /mnt/gestalt/home/ddmanddman/moisesdb/npyq

# python npyify.py get_query_from_onset --data_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2 \
#                     --query_root /mnt/gestalt/home/ddmanddman/moisesdb/npyq
# python npyify.py get_durations --data_root /mnt/gestalt/home/ddmanddman/moisesdb/npy2
# python npyify.py make_test_indices --metadata_path /mnt/gestalt/home/ddmanddman/moisesdb/metadata.csv \
#                     --stem_path /mnt/gestalt/home/ddmanddman/moisesdb/stems.csv \
#                     --splits_path /mnt/gestalt/home/ddmanddman/moisesdb/splits.csv


python npyify.py get_dbfs_by_chunk --data_root /home/buffett/NAS_NTU/moisesdb/npy2 \
                --query_root /home/buffett/NAS_NTU/moisesdb/npyq6

# python npyify.py get_query_from_onset --data_root /home/buffett/NAS_NTU/moisesdb/npy2 \
#                     --query_root /home/buffett/NAS_NTU/moisesdb/npyq6