import logging
import os


class Logger:
    """
    	Numeric value
        CRITICAL  50
        ERROR	  40
        WARNING	  30
        INFO	  20
        DEBUG	  10
        NOTSET    0
        
    USAGE:
    There are 3 ways you use this class:
    1) Logger().get_logger() -> will just push logging with DEBUG level into console
    2) Logger(level='INFO').get_logger() -> will push logging with INFO level into console
    3) Logger(file_name='aqt_file_name', level='INFO').get_logger() -> will push logging to both
    file (where the path will be formed as aqt_python2\log\aqt_file_name.txt) and also
    self.log_file_path is going to be filled in - which can be used in the mails

    TESTS:
    #myLogger1 = Logger() <-------- just a console DEBUG
    #myLogger1 = Logger(level='DEBUG') <------------ just a console DEBUG
    #myLogger1 = Logger(level='DEBUG', file_name='aqt_test_{}'.format(t)) <------- file into DEBUG console into DEBUG
    #myLogger1 = Logger(level='INFO', file_name='aqt_test_{}'.format(t)) <------ file into INFO, console into INFO
    #myLogger1 = Logger(level='DEBUG') <------ file into DEBUG console into DEBUG
    
    """
    instance = None

    class _Logger:

        def __init__(self, file_name=None, folder=None, level=None):
            # print("Running _Logger constructor!")
            self.logger = logging.getLogger('')
            if level is None:
                self.level = 'INFO'
                self.logger.setLevel(self.level)
            else:
                self.level = level
                self.logger.setLevel(level)
            self.clean_up_non_file_loggers()
            self.log_file = file_name
            self.log_folder = folder
            self.log_file_path = None

        def get_logger(self):
            return self.logger

        def switch_off(self):
            self.level = 'CRITICAL'
            self.clean_up_non_file_loggers()
            if self.log_file:
                self.set_level_for_file_handler()
            # print(self.logger.level, logging.getLevelName(self.logger.level))

        def switch_on(self, level='INFO'):
            self.level = level
            self.clean_up_non_file_loggers()
            if self.log_file:
                self.set_level_for_file_handler()
            # print(self.logger.level, logging.getLevelName(self.logger.level))

        def clean_up_non_file_loggers(self):
            handlers = self.logger.handlers[:]
            for handler in handlers:
                if 'FileHandler' not in str(handler):
                    handler.close()
                    self.logger.removeHandler(handler)
            # put logger to console
            console = logging.StreamHandler()
            console.setLevel(self.level)
            self.logger.addHandler(console)
            self.logger.setLevel(self.level)

        def logger_to_file(self):
            self.remove_file_handler()
            if self.log_file is not None:
                self.log_file_path = os.path.join(self.log_folder,
                                                  self.log_file + '.txt')
                if os.path.exists(self.log_file_path):
                    try:
                        os.remove(self.log_file_path)
                    except:
                        pass
                file_handler = logging.FileHandler(self.log_file_path)
                file_handler.setLevel(self.level)
                self.logger.addHandler(file_handler)

        def remove_file_handler(self):
            for hndl in self.logger.handlers:
                if 'FileHandler' in str(hndl):
                    self.logger.removeHandler(hndl)

        def set_level_for_file_handler(self):
            file_hnld = list(
                filter(lambda hanlder: 'filehandler' in str(hanlder).lower(),
                       self.logger.handlers))[0]
            file_hnld.setLevel(self.level)

    def __init__(self, file_name=False, folder=r'C:\temp', level='INFO'):
        if not Logger.instance:
            # print("We don't have logger")
            Logger.instance = Logger._Logger(level=level)
            if file_name:
                Logger.instance = Logger._Logger(
                    file_name=file_name, folder=folder, level=level)
                Logger.instance.logger_to_file()
        else:
            # print("We already have logger")
            # if we've already instantiated the logger - it will be only console
            # from smartdata.utils.common import # print_attributes as pa
            # pa(Logger.instance, True)

            if logging.getLevelName(
                    Logger.instance.level) > logging.getLevelName(level):
                Logger.instance.level = level  # once we set DEBUG level - we'll keep it
            Logger.instance.clean_up_non_file_loggers()
            # #
            if file_name:  #we've been given a file name
                if Logger.instance.log_file is None:  # we already have a file
                    Logger.instance.log_file = file_name  # re-create file
                    Logger.instance.log_folder = folder  # re-create folder
                    Logger.instance.logger_to_file()  # create logger to file
                else:
                    Logger.instance.remove_file_handler()  # we remove the file
                    Logger.instance.log_file = file_name  # re-create file
                    Logger.instance.log_folder = folder  # re-create folder
                    Logger.instance.logger_to_file()  # create logger to file
            else:
                if Logger.instance.log_file is not None:
                    Logger.instance.set_level_for_file_handler()

                # change level for existing files

    def __getattr__(self, name):
        return getattr(Logger.instance, name)
