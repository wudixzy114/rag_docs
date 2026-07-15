# 服务启动超时如何处理

当服务启动超过 60 秒仍未就绪时，先执行 `systemctl status demo.service` 查看状态，
再执行 `journalctl -u demo.service --since "10 minutes ago"` 检查最近日志。

如果日志出现 `address already in use`，使用 `ss -lntp | grep 8080` 定位占用 8080
端口的进程。确认该进程可以停止后先终止它，再重新启动 `demo.service`，最后通过
`curl http://127.0.0.1:8080/health` 验证健康检查返回成功。
