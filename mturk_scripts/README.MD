Run the Matlab code to generate the html (output_timed.txt) and csv data file (indicating the image URLs) needed for mturk. Read the code to understand the process.

example.html includes the rest of the html, javascript, and css needed to be copy-pasted to the "Source" view in the "Design Layout" page of the mturk project GUI. When you create your output_timed.txt html, copy and paste its contents between the "<!-- OUTPUT_TIMED.TXT CONTENT STARTS HERE -->" comments in example.html. There's some flavour text added to the first three rows for warmup that will need to be manually put in (check out example.html for an example).

To avoid the same worker working multiple HITS, use https://uniqueturker.myleott.com/. Put the generated uniqueturker ID in the ut_id variable at the top of example.html.
