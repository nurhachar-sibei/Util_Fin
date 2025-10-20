import logging
from datetime import datetime

def setup_logger(log_file: str = "RP_model",file_load = './log', level=logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    :param name: 日志记录器名称
    :param log_file: 日志文件路径
    :param level: 日志级别
    :return: 配置好的日志记录器
    """
    date_current = datetime.now().strftime("%Y%m%d")
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{file_load}/{log_file}_{date_current}.log",encoding="utf-8")
        ]
    )
    import inspect
    import os
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename 

    logger = logging.getLogger(os.path.basename(caller_file)[:-3])
    return logger

if __name__ == "__main__":
    logger = setup_logger("test",'.')
    logger.info("这是一条info日志2")
