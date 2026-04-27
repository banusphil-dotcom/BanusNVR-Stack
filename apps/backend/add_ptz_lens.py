"""Add C246D PTZ lens as a separate camera in the DB."""
import asyncio
from sqlalchemy import select, func
from models.database import async_session
from models.schemas import Camera, CameraType, RecordingMode

async def main():
    async with async_session() as session:
        # Show current cameras
        result = await session.execute(select(Camera).order_by(Camera.id))
        cameras = result.scalars().all()
        print("Current cameras:")
        for c in cameras:
            print(f"  camera_{c.id}: {c.name} ({c.camera_type.value})")
        
        max_id = max(c.id for c in cameras) if cameras else 0
        print(f"\nMax ID: {max_id}")
        
        # Check if PTZ lens already exists
        for c in cameras:
            if "PTZ" in c.name or "ptz" in c.name:
                print(f"\nPTZ camera already exists: camera_{c.id} ({c.name})")
                return
        
        # Add PTZ lens camera
        ptz_camera = Camera(
            name="Garden PTZ",
            camera_type=CameraType.tapo,
            connection_config={
                "ip": "192.168.68.116",
                "username": "banusnvr",
                "password": "Twins2021",
                "port": "554",
                "stream_path": "stream6",
                "sub_stream_path": "stream7",
            },
            recording_mode=RecordingMode.motion,
            detection_enabled=True,
            detection_objects=["person", "cat", "dog", "car"],
            detection_confidence=0.5,
            enabled=True,
            ptz_mode="onvif",
            ptz_config={
                "host": "192.168.68.116",
                "port": 2020,
                "user": "banusnvr",
                "password": "Twins2021",
            },
        )
        session.add(ptz_camera)
        await session.commit()
        await session.refresh(ptz_camera)
        print(f"\nAdded: camera_{ptz_camera.id}: {ptz_camera.name}")
        print(f"Use camera_{ptz_camera.id} in Frigate config")

asyncio.run(main())
