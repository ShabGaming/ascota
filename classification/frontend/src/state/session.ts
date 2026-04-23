import { create } from 'zustand'

export type ClassificationType = 'pottery' | 'type' | 'decoration' | 'color' | 'texture'

interface SessionState {
  sessionId: string | null
  contextPath: string | null
  classificationType: ClassificationType | null
  setSession: (sessionId: string, contextPath: string, classificationType: ClassificationType) => void
  reset: () => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  contextPath: null,
  classificationType: null,

  setSession: (sessionId: string, contextPath: string, classificationType: ClassificationType) => {
    set({ sessionId, contextPath, classificationType })
  },

  reset: () => {
    set({ sessionId: null, contextPath: null, classificationType: null })
  },
}))
