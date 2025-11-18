"""
PostgreSQL数据管理系统
用于简单的数据存储和管理

Author: Nurhachar
Date: 2025
"""

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import json
import logging
import time
from functools import wraps
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
import warnings

print("Easy Manager is running...")
# 配置日志
# 配置日志格式，使其更符合用户要求的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('datadeal.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def function_timer(func):
    """
    函数计时装饰器
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info(f'[Function: {func.__name__} started...]')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f'[Function: {func.__name__} completed, elapsed time: {elapsed_time:.2f}s]')
            return result
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f'[Function: {func.__name__} failed after {elapsed_time:.2f}s, error: {str(e)}]')
            raise
    
    return wrapper


class EasyManager:
    """
    简易PostgreSQL数据管理类
    支持创建表格、插入数据（去重）、删除表格、导入表格等操作
    """
    
    def __init__(self, 
                 database: str = "test_data_base",
                 user: str = "postgres", 
                 password: str = "cbw88982449",
                 host: str = "localhost",
                 port: str = "5432"):
        """
        初始化数据库连接
        
        Args:
            database: 数据库名
            user: 用户名
            password: 密码
            host: 主机地址
            port: 端口号
        """
        self.logger = logging.getLogger(__name__)
        self.db_config = {
            'database': database,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.logger.info(f"数据库连接成功: {self.db_config['database']}")
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """确保数据库连接有效"""
        try:
            self.cursor.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            self.logger.warning("数据库连接已断开，正在重新连接...")
            self._connect()
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """
        推断pandas列的PostgreSQL数据类型
        
        Args:
            series: pandas Series
            
        Returns:
            PostgreSQL数据类型字符串
        """
        dtype = series.dtype
        
        # 数值类型
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            return "DOUBLE PRECISION"
        # 布尔类型
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        # 日期时间类型
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        # 字符串类型
        else:
            # 检查最大长度
            max_length = series.astype(str).str.len().max()
            if pd.isna(max_length) or max_length == 0:
                return "TEXT"
            elif max_length <= 255:
                return f"VARCHAR({int(max_length * 1.5)})"  # 留点余量
            else:
                return "TEXT"
    
    @function_timer
    def create_table(self, table_name: str, dataframe: pd.DataFrame, 
                     overwrite: bool = False) -> bool:
        """
        在数据库中创建表格
        
        Args:
            table_name: 表名
            dataframe: pandas DataFrame
            overwrite: 是否覆盖已存在的表
            
        Returns:
            bool: 创建是否成功
        """
        self._ensure_connection()
        
        try:
            # 检查表是否存在
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name.split('.')[-1],))
            
            table_exists = self.cursor.fetchone()['exists']
            
            if table_exists and not overwrite:
                self.logger.warning(f"表 {table_name} 已存在，使用overwrite=True来覆盖")
                return False
            
            if table_exists and overwrite:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                self.logger.info(f"已删除现有表 {table_name}")
            
            # 创建表结构
            df = dataframe.copy()
            
            # 处理索引：如果索引有名称且不是默认的RangeIndex，将其作为列
            if df.index.name and not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()
            elif not isinstance(df.index, pd.RangeIndex):
                df.index.name = 'index'
                df = df.reset_index()
            
            # 构建列定义
            columns_sql = []
            for col in df.columns:
                col_type = self._infer_column_type(df[col])
                # 清理列名，确保符合SQL标准
                clean_col = col.replace('.', '_').replace('-', '_').replace(' ', '_')
                columns_sql.append(f'"{clean_col}" {col_type}')
            
            create_sql = f"""
                CREATE TABLE {table_name} (
                    {', '.join(columns_sql)}
                )
            """
            
            self.cursor.execute(create_sql)
            self.conn.commit()
            
            self.logger.info(f"表 {table_name} 创建成功，包含 {len(df.columns)} 列")
            
            # 插入数据
            self._insert_dataframe(table_name, df)
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            import traceback
            self.logger.error(f"创建表 {table_name} 失败: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def _insert_dataframe(self, table_name: str, df: pd.DataFrame):
        """
        将DataFrame插入到表中
        
        Args:
            table_name: 表名
            df: 要插入的DataFrame
        """
        if df.empty:
            self.logger.warning("DataFrame为空，跳过插入")
            return
        
        # 清理列名
        df_clean = df.copy()
        df_clean.columns = [col.replace('.', '_').replace('-', '_').replace(' ', '_') 
                           for col in df.columns]
        
        # 准备数据
        columns = ', '.join([f'"{col}"' for col in df_clean.columns])
        placeholders = ', '.join(['%s'] * len(df_clean.columns))
        
        insert_sql = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
        """
        
        # 转换数据为元组列表
        data_tuples = []
        for _, row in df_clean.iterrows():
            # 处理NaN值
            row_data = tuple(None if pd.isna(x) else x for x in row)
            data_tuples.append(row_data)
        
        # 批量插入
        psycopg2.extras.execute_batch(
            self.cursor, insert_sql, data_tuples, page_size=1000
        )
        self.conn.commit()
        
        self.logger.info(f"成功插入 {len(data_tuples)} 行数据到表 {table_name}")
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """
        获取表的列名列表
        
        Args:
            table_name: 表名
            
        Returns:
            列名列表
        """
        self.cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name.split('.')[-1],))
        
        return [row['column_name'] for row in self.cursor.fetchall()]
    
    @function_timer
    def add_columns(self, table_name: str, dataframe: pd.DataFrame, 
                    merge_on_index: bool = True) -> bool:
        """
        在表中增加新列，按索引合并数据
        
        Args:
            table_name: 表名
            dataframe: 包含新列的 DataFrame
            merge_on_index: 是否基于索引合并（默认True）
            
        Returns:
            bool: 添加是否成功
        """
        self._ensure_connection()
        
        try:
            # 检查表是否存在
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name.split('.')[-1],))
            
            if not self.cursor.fetchone()['exists']:
                self.logger.error(f"表 {table_name} 不存在")
                return False
            
            # 获取现有列
            existing_columns = self._get_table_columns(table_name)
            self.logger.info(f"表 {table_name} 现有列: {existing_columns}")
            
            # 处理DataFrame
            df = dataframe.copy()
            
            # 处理索引
            index_name = None
            if df.index.name and not isinstance(df.index, pd.RangeIndex):
                index_name = df.index.name
                df = df.reset_index()
            elif not isinstance(df.index, pd.RangeIndex):
                index_name = 'index'
                df.index.name = index_name
                df = df.reset_index()
            
            # 清理列名
            df.columns = [col.replace('.', '_').replace('-', '_').replace(' ', '_') 
                         for col in df.columns]
            if index_name:
                index_name = index_name.replace('.', '_').replace('-', '_').replace(' ', '_')
            
            # 识别新列（排除已存在的列）
            new_columns = [col for col in df.columns if col not in existing_columns]
            
            if not new_columns:
                self.logger.warning(f"没有新列需要添加到表 {table_name}")
                return True
            
            self.logger.info(f"识别到 {len(new_columns)} 个新列: {new_columns}")
            
            # 添加新列到表结构
            for col in new_columns:
                col_type = self._infer_column_type(df[col])
                alter_sql = f'ALTER TABLE {table_name} ADD COLUMN "{col}" {col_type}'
                self.cursor.execute(alter_sql)
                self.logger.info(f"添加列 {col} (类型: {col_type})")
            
            self.conn.commit()
            
            # 按索引合并数据
            if merge_on_index and index_name:
                # 加载现有数据
                existing_df = self.load_table(table_name)
                
                if existing_df is not None and not existing_df.empty:
                    self.logger.info(f"按索引列 '{index_name}' 合并数据")
                    
                    # 只保留新列和索引列
                    df_new_cols = df[[index_name] + new_columns]
                    
                    # 更新每一行的新列数据
                    update_count = 0
                    for _, row in df_new_cols.iterrows():
                        index_value = row[index_name]
                        
                        # 构建UPDATE语句
                        set_clause = ', '.join([f'"{col}" = %s' for col in new_columns])
                        update_sql = f"""
                            UPDATE {table_name}
                            SET {set_clause}
                            WHERE "{index_name}" = %s
                        """
                        
                        # 准备参数
                        values = [None if pd.isna(row[col]) else row[col] for col in new_columns]
                        values.append(index_value)
                        print("%%%%")
                        print(values)
                        print(update_sql)
                        self.cursor.execute(update_sql, values)
                        if self.cursor.rowcount > 0:
                            update_count += 1
                    
                    self.conn.commit()
                    self.logger.info(f"成功更新 {update_count} 行的新列数据")
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            import traceback
            self.logger.error(f"添加列到表 {table_name} 失败: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    @function_timer
    def insert_data(self, table_name: str, dataframe: pd.DataFrame, 
                    mode: str = 'skip') -> bool:
        """
        向表中插入数据，支持多种重复数据处理模式
        
        Args:
            table_name: 表名
            dataframe: pandas DataFrame
            mode: 重复数据处理模式
                - 'skip': 忽略重复数据，只插入新数据（默认）
                - 'update': 覆盖重复数据，基于索引更新
                - 'append': 直接追加，不检查重复
            
        Returns:
            bool: 插入是否成功
        """
        self._ensure_connection()
        
        if mode not in ['skip', 'update', 'append']:
            self.logger.error(f"不支持的模式: {mode}，请使用 'skip', 'update' 或 'append'")
            return False
        
        try:
            # 检查表是否存在
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name.split('.')[-1],))
            
            if not self.cursor.fetchone()['exists']:
                self.logger.error(f"表 {table_name} 不存在，请先使用create_table创建表")
                return False
            
            df = dataframe.copy()
            
            # 处理索引
            index_name = None
            if df.index.name and not isinstance(df.index, pd.RangeIndex):
                index_name = df.index.name
                df = df.reset_index()
            elif not isinstance(df.index, pd.RangeIndex):
                index_name = 'index'
                df.index.name = index_name
                df = df.reset_index()
            
            # 清理列名
            df.columns = [col.replace('.', '_').replace('-', '_').replace(' ', '_') 
                         for col in df.columns]
            if index_name:
                index_name = index_name.replace('.', '_').replace('-', '_').replace(' ', '_')
            
            # 根据模式处理数据
            if mode == 'append':
                # 直接插入，不检查重复
                self.logger.info(f"使用 append 模式，直接插入 {len(df)} 行数据")
                self._insert_dataframe(table_name, df)
                
            elif mode == 'skip':
                # 忽略重复索引的数据，只插入新索引的数据
                if not index_name:
                    self.logger.error("skip 模式需要有索引列，但未找到索引")
                    return False
                
                existing_df = self.load_table(table_name)
                
                if existing_df is not None and not existing_df.empty:
                    # 检查索引列是否存在
                    if index_name not in existing_df.columns:
                        self.logger.error(f"索引列 '{index_name}' 不存在于表中")
                        return False
                    
                    # 基于索引找出不重复的行
                    existing_indices = set(existing_df[index_name].values)
                    new_indices = set(df[index_name].values)
                    
                    # 只保留索引不重复的行
                    indices_to_insert = new_indices - existing_indices
                    
                    # 如果没有新数据
                    if not indices_to_insert:
                        self.logger.info(f"没有新的索引数据需要插入到表 {table_name}")
                        return True
                    
                    # 过滤出要插入的数据
                    df_to_insert = df[df[index_name].isin(indices_to_insert)]
                    
                    self.logger.info(f"使用 skip 模式，基于索引过滤后有 {len(df_to_insert)} 行新数据")
                    self._insert_dataframe(table_name, df_to_insert)
                else:
                    self.logger.info(f"表为空，插入 {len(df)} 行数据")
                    self._insert_dataframe(table_name, df)
                    
            elif mode == 'update':
                # 覆盖重复数据，基于索引更新
                if not index_name:
                    self.logger.error("update 模式需要有索引列，但未找到索引")
                    return False
                
                existing_df = self.load_table(table_name)
                
                if existing_df is None or existing_df.empty:
                    self.logger.info(f"表为空，直接插入 {len(df)} 行数据")
                    self._insert_dataframe(table_name, df)
                else:
                    # 检查索引列是否存在
                    if index_name not in existing_df.columns:
                        self.logger.error(f"索引列 '{index_name}' 不存在于表中")
                        return False
                    
                    # 获取所有列（排除索引列）
                    data_columns = [col for col in df.columns if col != index_name]
                    
                    # 找出需要更新的行和需要插入的行
                    existing_indices = set(existing_df[index_name].values)
                    new_indices = set(df[index_name].values)
                    
                    indices_to_update = existing_indices & new_indices
                    indices_to_insert = new_indices - existing_indices
                    
                    update_count = 0
                    insert_count = 0
                    
                    # 更新重复的行
                    if indices_to_update:
                        self.logger.info(f"使用 update 模式，更新 {len(indices_to_update)} 行")
                        df_to_update = df[df[index_name].isin(indices_to_update)]
                        
                        for _, row in df_to_update.iterrows():
                            index_value = row[index_name]
                            
                            # 构建UPDATE语句
                            set_clause = ', '.join([f'"{col}" = %s' for col in data_columns])
                            update_sql = f"""
                                UPDATE {table_name}
                                SET {set_clause}
                                WHERE "{index_name}" = %s
                            """
                            
                            # 准备参数
                            values = [None if pd.isna(row[col]) else row[col] for col in data_columns]
                            values.append(index_value)
                            
                            self.cursor.execute(update_sql, values)
                            if self.cursor.rowcount > 0:
                                update_count += 1
                        
                        self.conn.commit()
                        self.logger.info(f"成功更新 {update_count} 行数据")
                    
                    # 插入新的行
                    if indices_to_insert:
                        self.logger.info(f"插入 {len(indices_to_insert)} 行新数据")
                        df_to_insert = df[df[index_name].isin(indices_to_insert)]
                        self._insert_dataframe(table_name, df_to_insert)
                        insert_count = len(df_to_insert)
                    
                    self.logger.info(f"update 模式完成: 更新 {update_count} 行, 插入 {insert_count} 行")
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            import traceback
            self.logger.error(f"插入数据到表 {table_name} 失败: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    @function_timer
    def drop_table(self, table_name: str) -> bool:
        """
        删除表格
        
        Args:
            table_name: 表名
            
        Returns:
            bool: 删除是否成功
        """
        self._ensure_connection()
        
        try:
            # 检查表是否存在
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name.split('.')[-1],))
            
            if not self.cursor.fetchone()['exists']:
                self.logger.warning(f"表 {table_name} 不存在")
                return False
            
            # 删除表
            self.cursor.execute(f"DROP TABLE {table_name} CASCADE")
            self.conn.commit()
            
            self.logger.info(f"成功删除表: {table_name}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            import traceback
            self.logger.error(f"删除表 {table_name} 失败: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    @function_timer
    def load_table(self, table_name: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        从数据库导入表格到Python
        
        Args:
            table_name: 表名
            limit: 限制返回行数（可选）
            
        Returns:
            pandas DataFrame 或 None
        """
        self._ensure_connection()
        
        try:
            # 检查表是否存在
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name.split('.')[-1],))
            
            if not self.cursor.fetchone()['exists']:
                self.logger.error(f"表 {table_name} 不存在")
                return None
            
            # 构建查询
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            # 执行查询
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            if not results:
                self.logger.warning(f"表 {table_name} 为空")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            
            self.logger.info(f"成功从表 {table_name} 加载数据，形状: {df.shape}")
            return df
            
        except Exception as e:
            import traceback
            self.logger.error(f"加载表 {table_name} 失败: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def list_tables(self, schema: str = 'public') -> List[str]:
        """
        列出数据库中所有表
        
        Args:
            schema: 模式名（默认为public）
            
        Returns:
            表名列表
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            
            tables = [row['table_name'] for row in self.cursor.fetchall()]
            self.logger.info(f"找到 {len(tables)} 个表")
            return tables
            
        except Exception as e:
            self.logger.error(f"获取表列表失败: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        获取表信息
        
        Args:
            table_name: 表名
            
        Returns:
            表信息字典
        """
        self._ensure_connection()
        
        try:
            # 获取列信息
            self.cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name.split(".")[-1],))
            
            columns = self.cursor.fetchall()
            
            # 获取行数
            self.cursor.execute(f"SELECT COUNT(*) as row_count FROM {table_name}")
            row_count = self.cursor.fetchone()['row_count']
            
            info = {
                'table_name': table_name,
                'columns': columns,
                'row_count': row_count
            }
            
            self.logger.info(f"表 {table_name} 信息: {len(columns)} 列, {row_count} 行")
            return info
            
        except Exception as e:
            self.logger.error(f"获取表信息失败: {str(e)}")
            return {}
    
    @staticmethod
    def help():
        """
        显示 EasyManager 的功能帮助信息
        """
        help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      EasyManager 使用帮助 (v2.1)                           ║
╚════════════════════════════════════════════════════════════════════════════╝

📚 核心功能：

  1. create_table(table_name, dataframe, overwrite=False)
     └─ 创建表并导入数据
     └─ 示例: em.create_table('my_table', df, overwrite=True)

  2. add_columns(table_name, dataframe, merge_on_index=True)  ⭐ 新功能
     └─ 在已存在的表中添加新列（自动屏蔽已存在的列）
     └─ 示例: em.add_columns('my_table', df_new_cols)

  3. insert_data(table_name, dataframe, mode='skip')  ⭐ 升级版
     └─ 插入数据，支持三种模式：
        • skip   - 忽略重复索引（默认）
        • update - 覆盖重复索引的数据
        • append - 直接追加，不检查重复
     └─ 示例: em.insert_data('my_table', df, mode='skip')

  4. load_table(table_name, limit=None)
     └─ 从数据库导入表到 Python
     └─ 示例: df = em.load_table('my_table', limit=100)

  5. drop_table(table_name)
     └─ 删除表
     └─ 示例: em.drop_table('my_table')

  6. list_tables(schema='public')
     └─ 列出所有表
     └─ 示例: tables = em.list_tables()

  7. get_table_info(table_name)
     └─ 获取表详细信息（列、行数等）
     └─ 示例: info = em.get_table_info('my_table')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 三种插入模式详解：

  ┌─────────┬──────────┬──────────┬──────────┬───────────────────┐
  │  模式   │ 检查方式 │ 需要索引 │   性能   │     适用场景      │
  ├─────────┼──────────┼──────────┼──────────┼───────────────────┤
  │ skip    │ 基于索引 │   ✅     │   中等   │ 增量更新，避免重复│
  │ update  │ 基于索引 │   ✅     │   较慢   │ 数据修正，UPSERT  │
  │ append  │ 不检查   │   ❌     │   最快   │ 快速批量导入      │
  └─────────┴──────────┴──────────┴──────────┴───────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 快速示例：

  # 1. 连接数据库
  from easy_manager import EasyManager
  import pandas as pd
  
  with EasyManager() as em:
      
      # 2. 创建表
      df = pd.read_csv('data.csv', index_col=0)
      em.create_table('stocks', df)
      
      # 3. 添加新列
      df_new = pd.read_csv('new_factors.csv', index_col=0)
      em.add_columns('stocks', df_new)
      
      # 4. 插入数据（三种模式）
      em.insert_data('stocks', df, mode='skip')    # 忽略重复索引
      em.insert_data('stocks', df, mode='update')  # 覆盖重复数据
      em.insert_data('stocks', df, mode='append')  # 直接追加
      
      # 5. 导入表
      df_loaded = em.load_table('stocks')
      
      # 6. 查询表信息
      tables = em.list_tables()
      info = em.get_table_info('stocks')
      
      # 7. 删除表
      em.drop_table('stocks')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  注意事项：

  • skip 和 update 模式需要 DataFrame 有索引列
  • 列名中的特殊字符（., -, 空格）会自动转换为 _
  • 所有操作记录在 datadeal.log 文件中

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 更多信息：

  • 完整手册：EasyManager完整使用手册.md
  • 测试示例：test_new_features.py
  • 在线帮助：EasyManager.help()

╚════════════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.logger.info("数据库连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 使用示例
if __name__ == "__main__":
    # 创建管理器实例
    with EasyManager() as em:
        # 列出所有表
        tables = em.list_tables()
        print("现有表:", tables)