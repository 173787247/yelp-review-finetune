@echo off
echo ========================================
echo YelpReviewFull 完整数据集训练
echo 快速启动 Jupyter (使用现有镜像)
echo ========================================
echo.

echo 启动容器...
docker-compose up -d

if errorlevel 1 (
    echo 错误: 容器启动失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 容器启动成功！
echo ========================================
echo.
echo Jupyter Lab 地址: http://localhost:8890
echo.
echo 注意: 完整数据集训练需要较长时间（数小时）
echo 建议: 可以先运行一个配置测试，确认环境正常
echo.
echo 查看日志: docker-compose logs -f
echo 停止容器: docker-compose down
echo.
timeout /t 5
start http://localhost:8890
pause

