import os

def iter_line_ltd(file, uppercap, lowercap):
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            if len(line) <= uppercap and len(line) >= lowercap:
                yield line

def main(infile: str, outpath: str, line_uppercap: int, line_lowercap: int, max_n_lines: int):
    outlines = []
    for i,line in enumerate(iter_line_ltd(infile,line_uppercap, line_lowercap),1):
        outlines.append(line)
        if i >= max_n_lines:
            break
    outfile = os.path.join(outpath, f'MONO-en-TED-{max_n_lines}.txt')
    with open(outfile, 'w', encoding='utf8') as f:
        f.writelines(outlines)
    
    print(f"Number of lines shorter than {line_uppercap}: ", len(outlines))
    
    assert len(outlines) == max_n_lines, f"The input file does not contain {max_n_lines} lines shorter than {line_uppercap} chars. Try increasing the max_line_length."


if __name__ == "__main__":
    main(
        infile="..\en-TED.txt",
        outpath="data\monolingual",
        line_uppercap=50,
        line_lowercap=20,
        max_n_lines=20000
    )

