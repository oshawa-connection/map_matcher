import subprocess

from extent import Extent
def runMapserverSubprocess(outputFilename:str, extent: Extent,imageSizePixels:int):

    x=extent.toArguments()
    starting_run_result = subprocess.run([ 
        '/hello/MapServer/build/map2img', 
        '-s', 
        str(imageSizePixels), 
        str(imageSizePixels), 
        '-all_debug', 
        '5', 
        '-map_debug', 
        '5', 
        '-l', 
        'buildings', 
        '-m', 
        '/mapfiles/some.map', 
        '-conf', 
        '/mapfiles/config.map', 
        '-e',
        *x,
        '-o',
        outputFilename ],stdout = subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    starting_run_result.check_returncode()