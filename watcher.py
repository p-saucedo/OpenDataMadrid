#!/usr/bin/python3
import logging

def get_logger(name):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename='activity.log',
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    
    return logging.getLogger(name)
