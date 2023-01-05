import os
for f in os.listdir('./static/images'):
    new_name = f.replace(' ' , '')
    if f != new_name:
        os.rename('./static/images/' + f, './static/images/' + new_name)
 
