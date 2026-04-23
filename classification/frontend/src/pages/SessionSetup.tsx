import { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Container,
  FormControl,
  FormLabel,
  Heading,
  Input,
  VStack,
  HStack,
  IconButton,
  useToast,
  Card,
  CardBody,
  Text,
  Link,
  Radio,
  RadioGroup,
  Stack,
  Tooltip,
  List,
  ListItem,
  Badge,
} from '@chakra-ui/react'
import { DeleteIcon } from '@chakra-ui/icons'
import {
  createSession,
  checkContextStatus,
  checkPotteryGate,
  listSessions,
  loadSession,
  deleteSession,
  type SessionSummary,
  type PotteryGateStatus,
} from '../api/client'
import { useSessionStore } from '../state/session'
import type { ClassificationType } from '../state/session'

interface SessionSetupProps {
  onComplete: () => void
}

function mapSessionClassificationType(t: string): ClassificationType {
  if (t === 'pottery') return 'pottery'
  if (t === 'decoration') return 'decoration'
  if (t === 'color') return 'color'
  if (t === 'texture') return 'texture'
  return 'type'
}

function SessionSetup({ onComplete }: SessionSetupProps) {
  const [contextPath, setContextPath] = useState('')
  const [isPreprocessed, setIsPreprocessed] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [classificationType, setClassificationType] = useState<string>('pottery')
  const [checkingPath, setCheckingPath] = useState(false)
  const [potteryGate, setPotteryGate] = useState<PotteryGateStatus | null>(null)
  const [gateLoading, setGateLoading] = useState(false)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const toast = useToast()
  const setSession = useSessionStore((s) => s.setSession)

  const refreshSessions = async () => {
    setLoadingSessions(true)
    try {
      const data = await listSessions()
      setSessions(data.sessions || [])
    } catch {
      setSessions([])
    } finally {
      setLoadingSessions(false)
    }
  }

  useEffect(() => {
    refreshSessions()
  }, [])

  const categoryGateOk =
    isPreprocessed && !gateLoading && Boolean(potteryGate?.complete)

  useEffect(() => {
    const trimmed = contextPath.trim()
    if (!trimmed || !isPreprocessed) {
      setPotteryGate(null)
      setGateLoading(false)
      return
    }
    setGateLoading(true)
    checkPotteryGate(trimmed)
      .then(setPotteryGate)
      .catch(() => setPotteryGate(null))
      .finally(() => setGateLoading(false))
  }, [contextPath, isPreprocessed])

  const handleLoadSession = async (s: SessionSummary) => {
    setLoadingId(s.session_id)
    try {
      const data = await loadSession(s.session_id)
      setSession(data.session_id, data.context_path, mapSessionClassificationType(data.classification_type))
      toast({
        title: 'Session loaded',
        description: data.has_results ? 'Results loaded.' : 'No results yet – run classification.',
        status: 'success',
        duration: 2000,
      })
      onComplete()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load session'
      toast({ title: 'Load failed', description: msg, status: 'error', duration: 4000 })
    } finally {
      setLoadingId(null)
    }
  }

  const handleDeleteSession = async (sessionId: string) => {
    setDeletingId(sessionId)
    try {
      await deleteSession(sessionId)
      await refreshSessions()
      toast({ title: 'Session deleted', status: 'success', duration: 2000 })
    } catch {
      toast({ title: 'Delete failed', status: 'error', duration: 3000 })
    } finally {
      setDeletingId(null)
    }
  }

  const handleCheckContext = async () => {
    const trimmed = contextPath.trim()
    if (!trimmed) return
    setCheckingPath(true)
    try {
      const status = await checkContextStatus(trimmed)
      setIsPreprocessed(status.is_preprocessed)
      if (status.is_preprocessed) {
        toast({ title: 'Context is preprocessed', status: 'success', duration: 2000 })
      } else {
        toast({
          title: 'Context not preprocessed',
          description: 'Run the preprocess pipeline on this context first.',
          status: 'warning',
          duration: 5000,
        })
      }
    } catch {
      toast({
        title: 'Could not check context',
        description: 'Path may be invalid or backend unreachable.',
        status: 'error',
        duration: 4000,
      })
      setIsPreprocessed(false)
    } finally {
      setCheckingPath(false)
    }
  }

  const handleStart = async () => {
    const trimmed = contextPath.trim()
    if (!trimmed) {
      toast({ title: 'Enter a context path', status: 'warning', duration: 3000 })
      return
    }
    if (!isPreprocessed) {
      toast({
        title: 'Context must be preprocessed',
        description: 'Check context and ensure it shows as preprocessed.',
        status: 'warning',
        duration: 4000,
      })
      return
    }
    const allowed =
      classificationType === 'pottery' ||
      classificationType === 'type' ||
      classificationType === 'decoration' ||
      classificationType === 'color' ||
      classificationType === 'texture'
    if (!allowed) {
      toast({ title: 'Select a classification option', status: 'info', duration: 3000 })
      return
    }
    if (
      (classificationType === 'type' ||
        classificationType === 'decoration' ||
        classificationType === 'color' ||
        classificationType === 'texture') &&
      !categoryGateOk
    ) {
      toast({
        title: 'Pottery gate incomplete',
        description: 'Export pottery vs non-pottery for every image in this context first.',
        status: 'warning',
        duration: 5000,
      })
      return
    }
    setIsLoading(true)
    try {
      const response = await createSession(trimmed, classificationType)
      setSession(response.session_id, trimmed, mapSessionClassificationType(classificationType))
      toast({
        title: 'Session created',
        description: `${response.items_count} images to classify.`,
        status: 'success',
        duration: 2000,
      })
      onComplete()
    } catch (error: unknown) {
      let message = 'Unknown error'
      if (error && typeof error === 'object' && 'response' in error) {
        const d = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
        if (typeof d === 'string') message = d
      } else if (error instanceof Error) {
        message = error.message
      }
      toast({ title: 'Failed to create session', description: message, status: 'error', duration: 5000 })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Container maxW="container.lg" py={10}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading size="xl" mb={2}>
            Classification Pipeline
          </Heading>
          <Text color="gray.600">Run classification on a preprocessed context.</Text>
        </Box>

        <Card>
          <CardBody>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel>Context directory</FormLabel>
                <HStack>
                  <Input
                    placeholder="D:\ararat\data\files\N\38\478020\4419550\1"
                    value={contextPath}
                    onChange={(e) => setContextPath(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCheckContext()}
                  />
                  <Button
                    size="md"
                    colorScheme="gray"
                    onClick={handleCheckContext}
                    isLoading={checkingPath}
                  >
                    Check
                  </Button>
                </HStack>
                {contextPath.trim() && (
                  <Text mt={2} fontSize="sm" color={isPreprocessed ? 'green.600' : 'orange.600'}>
                    {isPreprocessed ? 'Preprocessed – you can continue.' : 'Not preprocessed – run preprocess first.'}
                  </Text>
                )}
              </FormControl>

              <FormControl>
                <FormLabel>Classification</FormLabel>
                <RadioGroup value={classificationType} onChange={setClassificationType}>
                  <VStack align="stretch" spacing={4}>
                    <Box>
                      <Text fontSize="sm" fontWeight="semibold" color="gray.600" mb={2}>
                        Pottery
                      </Text>
                      <Radio value="pottery">Pottery vs non-pottery</Radio>
                    </Box>
                    <Box>
                      <Text fontSize="sm" fontWeight="semibold" color="gray.600" mb={2}>
                        Categories (pottery images only)
                      </Text>
                      {isPreprocessed && gateLoading && (
                        <Text fontSize="xs" color="gray.500" mb={2}>
                          Checking pottery labels on disk…
                        </Text>
                      )}
                      {isPreprocessed && !gateLoading && potteryGate && !potteryGate.complete && (
                        <Text fontSize="xs" color="orange.600" mb={2}>
                          Export pottery first: {potteryGate.missing_count} of {potteryGate.total} images missing
                          on-disk labels.
                        </Text>
                      )}
                      {isPreprocessed && !gateLoading && potteryGate?.complete && (
                        <Text fontSize="xs" color="green.600" mb={2}>
                          Pottery gate complete ({potteryGate.pottery_on_disk_count} pottery image
                          {potteryGate.pottery_on_disk_count === 1 ? '' : 's'} on disk).
                        </Text>
                      )}
                      <Stack direction="row" spacing={4} flexWrap="wrap">
                        <Tooltip
                          label={
                            categoryGateOk ? undefined : 'Run and export Pottery vs non-pottery for this context first'
                          }
                          hasArrow
                        >
                          <Box>
                            <Radio value="type" isDisabled={!categoryGateOk}>
                              Type
                            </Radio>
                          </Box>
                        </Tooltip>
                        <Tooltip
                          label={
                            categoryGateOk ? undefined : 'Run and export Pottery vs non-pottery for this context first'
                          }
                          hasArrow
                        >
                          <Box>
                            <Radio value="decoration" isDisabled={!categoryGateOk}>
                              Decoration / Pattern
                            </Radio>
                          </Box>
                        </Tooltip>
                        <Tooltip
                          label={
                            categoryGateOk ? undefined : 'Run and export Pottery vs non-pottery for this context first'
                          }
                          hasArrow
                        >
                          <Box>
                            <Radio value="color" isDisabled={!categoryGateOk}>
                              Color
                            </Radio>
                          </Box>
                        </Tooltip>
                        <Tooltip
                          label={
                            categoryGateOk ? undefined : 'Run and export Pottery vs non-pottery for this context first'
                          }
                          hasArrow
                        >
                          <Box>
                            <Radio value="texture" isDisabled={!categoryGateOk}>
                              Texture
                            </Radio>
                          </Box>
                        </Tooltip>
                      </Stack>
                    </Box>
                  </VStack>
                </RadioGroup>
              </FormControl>
            </VStack>
          </CardBody>
        </Card>

        <Button
          colorScheme="brand"
          size="lg"
          onClick={handleStart}
          isLoading={isLoading}
          loadingText="Creating session..."
          isDisabled={!contextPath.trim() || !isPreprocessed}
        >
          Continue
        </Button>

        {sessions.length > 0 && (
          <Card w="100%">
            <CardBody>
              <Heading size="sm" mb={3}>
                Existing sessions
              </Heading>
              <Text fontSize="sm" color="gray.600" mb={3}>
                Load a session to continue or export; only unclassified images will run when you click Run.
              </Text>
              {loadingSessions ? (
                <Text fontSize="sm" color="gray.500">Loading…</Text>
              ) : (
                <List spacing={2}>
                  {sessions.map((s) => (
                    <ListItem
                      key={s.session_id}
                      p={2}
                      bg="gray.50"
                      borderRadius="md"
                      display="flex"
                      flexWrap="wrap"
                      alignItems="center"
                      justifyContent="space-between"
                      gap={2}
                    >
                      <Box flex={1} minW={0}>
                        <Text fontSize="sm" fontFamily="mono" noOfLines={1} title={s.context_path}>
                          {s.context_path}
                        </Text>
                        <HStack mt={1} spacing={2}>
                          <Badge size="sm" colorScheme="blue">
                            {s.classification_type}
                          </Badge>
                          <Text fontSize="xs" color="gray.500">
                            {s.results_count ?? 0} / {s.items_count} classified
                          </Text>
                        </HStack>
                      </Box>
                      <HStack>
                        <Button
                          size="sm"
                          colorScheme="brand"
                          onClick={() => handleLoadSession(s)}
                          isLoading={loadingId === s.session_id}
                        >
                          Load
                        </Button>
                        <IconButton
                          aria-label="Delete session"
                          icon={<DeleteIcon />}
                          size="sm"
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleDeleteSession(s.session_id)}
                          isLoading={deletingId === s.session_id}
                        />
                      </HStack>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardBody>
          </Card>
        )}

        <Box mt={8} pt={6} borderTop="1px" borderColor="gray.200">
          <VStack spacing={3}>
            <Text fontSize="sm" color="gray.600" textAlign="center">
              ASCOTA Classification Pipeline | APSAP
            </Text>
            <HStack spacing={4} justify="center" flexWrap="wrap">
              <Link
                href="https://github.com/ShabGaming/ascota"
                isExternal
                fontSize="sm"
                color="blue.500"
                _hover={{ color: 'blue.600', textDecoration: 'underline' }}
              >
                Repository
              </Link>
            </HStack>
          </VStack>
        </Box>
      </VStack>
    </Container>
  )
}

export default SessionSetup
