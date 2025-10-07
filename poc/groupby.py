import pandas as pd
import gc

def process_csv_in_chunks(input_file, output_file, chunk_size=100000):
    print(f"Processing {input_file} in chunks of {chunk_size} rows...")
    
    # Create a dictionary to store the grouped results
    grouped_data = {}
    
    # Process the CSV in chunks to avoid memory issues
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num+1}...")
        
        # Create a src_dst pair column
        chunk['src_dst_pair'] = chunk['IPV4_SRC_ADDR'] + '_' + chunk['IPV4_DST_ADDR']
        
        # Process each row in the chunk
        for _, row in chunk.iterrows():
            pair = row['src_dst_pair']
            
            if pair in grouped_data:
                # Update existing entry
                grouped_data[pair]['flow_count'] += 1
                grouped_data[pair]['total_distance'] += row['distance']
                grouped_data[pair]['total_demand'] += row['demand']
            else:
                # Create new entry
                grouped_data[pair] = {
                    'src_ip': row['IPV4_SRC_ADDR'],
                    'dst_ip': row['IPV4_DST_ADDR'],
                    'flow_count': 1,
                    'total_distance': row['distance'],
                    'total_demand': row['demand']
                }
        
        # Free memory
        del chunk
        gc.collect()
    
    # Convert the grouped data to DataFrame
    print("Creating final grouped DataFrame...")
    result_data = []
    for pair, data in grouped_data.items():
        result_data.append({
            'IPV4_SRC_ADDR': data['src_ip'],
            'IPV4_DST_ADDR': data['dst_ip'],
            'flow_count': data['flow_count'],
            'avg_distance': data['total_distance'] / data['flow_count'],
            'sum_demand': data['total_demand']
        })
    
    # Create and save the result DataFrame
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Grouped data saved to {output_file}")

# Usage
input_file = "netflow_with_distance_demand.csv"
output_file = "netflow_grouped_by_src_dst.csv"
process_csv_in_chunks(input_file, output_file)