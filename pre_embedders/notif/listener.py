import asyncio
import sqlite3
import time


db_conn = None
previous_notif_ids = set()


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect("data/notif_log.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_arrival REAL,
            timestamp_action REAL,
            app_name TEXT,
            notif_id INTEGER,
            interaction_type TEXT CHECK (interaction_type IN ('added', 'dismissed', 'expired')),
            response_time REAL
        )
        """
    )
    conn.commit()
    return conn


async def poll_notifications() -> None:
    global db_conn
    global previous_notif_ids
    try:
        from winsdk.windows.ui.notifications import NotificationKinds  # pyright: ignore[reportMissingImports]
        from winsdk.windows.ui.notifications.management import UserNotificationListener  # pyright: ignore[reportMissingImports]

        listener = UserNotificationListener.current
        notifications = await listener.get_notifications_async(NotificationKinds.TOAST)
        current_notifs = {
            notif.id: notif.app_info.display_info.display_name for notif in notifications
        }

        current_ids = set(current_notifs.keys())
        added_ids = current_ids - previous_notif_ids
        removed_ids = previous_notif_ids - current_ids

        for notif_id in added_ids:
            db_conn.execute(
                """
                INSERT INTO notifications (
                    timestamp_arrival,
                    timestamp_action,
                    app_name,
                    notif_id,
                    interaction_type,
                    response_time
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (time.time(), None, current_notifs[notif_id], notif_id, "added", None),
            )

        for notif_id in removed_ids:
            cursor = db_conn.execute(
                """
                SELECT id, timestamp_arrival
                FROM notifications
                WHERE notif_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (notif_id,),
            )
            row = cursor.fetchone()
            if row is None:
                continue

            row_id, timestamp_arrival = row
            timestamp_action = time.time()
            response_time = timestamp_action - timestamp_arrival

            db_conn.execute(
                """
                UPDATE notifications
                SET timestamp_action = ?,
                    interaction_type = ?,
                    response_time = ?
                WHERE id = ?
                """,
                (timestamp_action, "dismissed", response_time, row_id),
            )

        previous_notif_ids = current_ids
        db_conn.commit()
    except Exception as error:
        print(error)


async def run_listener() -> None:
    from winsdk.windows.ui.notifications.management import UserNotificationListener  # pyright: ignore[reportMissingImports]

    listener = UserNotificationListener.current
    access_status = await listener.request_access_async()
    print(access_status)

    while True:
        await poll_notifications()
        await asyncio.sleep(2)


if __name__ == "__main__":
    db_conn = init_db()
    print("Notification listener started. Press Ctrl+C to stop.")
    try:
        asyncio.run(run_listener())
    except KeyboardInterrupt:
        print("Stopping...")
        if db_conn is not None:
            db_conn.close()