"""方案二：Comp-ACT 模仿学习（独立包）。

隔离约束：对方案一代码(drivers/control/common/calib/perception/tasks/config)
只读 import、零修改；所有扩展经 il/adapters.py 包装实现。
配置在 il/config_il.yaml，数据/日志在 il/logs_il/。
"""
