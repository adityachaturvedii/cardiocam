import { useCallback, useEffect, useRef, useState } from 'react'

export type CameraState =
  | { status: 'idle' }
  | { status: 'requesting' }
  | { status: 'running'; stream: MediaStream }
  | { status: 'error'; error: string }

export interface UseCameraResult {
  state: CameraState
  devices: MediaDeviceInfo[]
  deviceId: string | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  start: (deviceId?: string) => Promise<void>
  stop: () => void
  setDeviceId: (id: string) => void
}

// A minimal wrapper around getUserMedia that: (1) keeps a MediaStream alive
// and attached to a <video>, (2) enumerates cameras once permission is
// granted (enumerateDevices returns blank labels before first permission),
// (3) lets the caller switch cameras without fighting lifecycle order.
export function useCamera(): UseCameraResult {
  const [state, setState] = useState<CameraState>({ status: 'idle' })
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setState({ status: 'idle' })
  }, [])

  const start = useCallback(async (preferredId?: string) => {
    // Clean up any previous stream before opening a new one — browsers will
    // otherwise return the same track or error with OverconstrainedError.
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    setState({ status: 'requesting' })
    try {
      const constraints: MediaStreamConstraints = {
        video: preferredId
          ? { deviceId: { exact: preferredId }, width: 1280, height: 720 }
          : { facingMode: 'user', width: 1280, height: 720 },
        audio: false,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => {
          // Autoplay can be rejected on some mobile browsers when the video
          // element is muted but not yet on-screen; caller can retry after
          // a user gesture.
        })
      }
      // Now that permission has been granted, device labels become populated.
      const all = await navigator.mediaDevices.enumerateDevices()
      const cams = all.filter((d) => d.kind === 'videoinput')
      setDevices(cams)
      const activeTrackSettings = stream.getVideoTracks()[0]?.getSettings()
      if (activeTrackSettings?.deviceId) {
        setDeviceId(activeTrackSettings.deviceId)
      }
      setState({ status: 'running', stream })
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Camera access failed.'
      setState({ status: 'error', error: message })
    }
  }, [])

  // Tear the stream down when the component unmounts.
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
    }
  }, [])

  return { state, devices, deviceId, videoRef, start, stop, setDeviceId }
}
