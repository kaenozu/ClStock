import time

from ClStock.systems.process_manager import ProcessManager


def test_monitoring_can_restart_after_stop():
    manager = ProcessManager()
    try:
        manager.start_monitoring()
        time.sleep(0.1)

        assert manager.monitor_thread is not None
        assert manager.monitor_thread.is_alive()

        manager.stop_monitoring()
        assert manager._shutdown_event.is_set()

        manager.start_monitoring()
        time.sleep(0.1)

        assert manager.monitor_thread is not None
        assert manager.monitor_thread.is_alive()
        assert not manager._shutdown_event.is_set()
    finally:
        manager.stop_monitoring()
        manager.wait_for_shutdown()
