class FileError(IOError):
    pass



try:
    raise FileError, "Test FileError"
except FileError,e:
    print e