import logging
from datetime import datetime

def setup_logger(log_file: str = "RP_model", file_load = './log', level=logging.INFO) -> logging.Logger:
    """
    设置日志记录器，支持多个独立的日志记录器
    :param log_file: 日志文件名（不含扩展名）
    :param file_load: 日志文件保存路径
    :param level: 日志级别
    :return: 配置好的日志记录器
    """
    date_current = datetime.now().strftime("%Y%m%d")
    name_ = f"{log_file}_{date_current}"
    
    # 获取或创建logger
    logger = logging.getLogger(name_)
    
    # 如果logger已经有handlers，说明已经配置过了，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 防止日志传播到父logger（避免重复记录）
    logger.propagate = False
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(
        f"{file_load}/{log_file}_{date_current}.log", 
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    import inspect
    import os
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename 
    # print(f"Logger '{name_}' 已创建，来自文件: {os.path.basename(caller_file)}")
    
    return logger

if __name__ == "__main__":
    logger_e = setup_logger("test",'.')
    logger_e.info("这是一条info日志2")
    logger_easy = setup_logger("datadeal",'.')
    logger_easy.info("这是一条info日志3")
