from logparser import Drain

input_dir = "dataset/"  # The input directory of log file
output_dir = "dataset/"  # The output directory of parsing results
log_file_all = "HDFS.log"  # The input log file name #!CHANGE
log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
regex = [
    r"blk_(|-)[0-9]+",  # block id
    r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP
    r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

# run on complete dataset
parser = Drain.LogParser(
    log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex
)
parser.parse(log_file_all)