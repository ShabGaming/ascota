import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  Box,
  Button,
  Container,
  Heading,
  VStack,
  HStack,
  Select,
  FormControl,
  FormLabel,
  useToast,
  SimpleGrid,
  Text,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Spinner,
  Flex,
  Tooltip,
  Divider,
  Switch,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from '@chakra-ui/react'
import {
  runPotteryClassification,
  getPotteryResults,
  getSession,
  updatePotteryResult,
  undoPotteryEdit,
  exportSession,
  getImageUrl,
  PotteryRunOptions,
  PotteryResultRow,
} from '../api/client'

const POTTERY_LABELS = ['pottery', 'non_pottery'] as const

function potteryLabelDisplay(label: string): string {
  if (label === 'pottery') return 'Pottery'
  if (label === 'non_pottery') return 'Non-pottery'
  return label
}

interface PotteryClassificationProps {
  sessionId: string
  onReset: () => void
}

interface ResultEntry {
  item_id: string
  label: string
  confidence: number
  p_pottery?: number
  find_number?: string
  image_filename?: string
}

function PotteryClassification({ sessionId, onReset }: PotteryClassificationProps) {
  const [options, setOptions] = useState<PotteryRunOptions>({ resolution: 1500 })
  const [results, setResults] = useState<Record<string, PotteryResultRow>>({})
  const [runLoading, setRunLoading] = useState(false)
  const [hasRun, setHasRun] = useState(false)
  const [sortAsc, setSortAsc] = useState(true)
  const [transparentMode, setTransparentMode] = useState(false)
  const [focusOnPottery, setFocusOnPottery] = useState(false)
  const [savingOverride, setSavingOverride] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState(false)
  const [viewMode, setViewMode] = useState<'review' | 'swipe'>('review')
  const [swipeThreshold, setSwipeThreshold] = useState(0.85)
  const swipeFocusRef = useRef<HTMLDivElement>(null)
  const toast = useToast()

  const fetchResults = async () => {
    try {
      const data = await getPotteryResults(sessionId)
      setResults(data.results || {})
    } catch {
      toast({ title: 'Failed to load results', status: 'error' })
    }
  }

  useEffect(() => {
    if (hasRun && sessionId) fetchResults()
  }, [sessionId, hasRun])

  useEffect(() => {
    if (!sessionId) return
    getSession(sessionId)
      .then((session) => {
        if (session.has_results) {
          setHasRun(true)
          return getPotteryResults(sessionId).then((data) => {
            if (data?.results) setResults(data.results)
          })
        }
      })
      .catch(() => {})
  }, [sessionId])

  const handleRun = async () => {
    setRunLoading(true)
    try {
      await runPotteryClassification(sessionId, options)
      setHasRun(true)
      await fetchResults()
      toast({ title: 'Classification complete', status: 'success', duration: 2000 })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Run failed'
      toast({ title: 'Classification failed', description: msg, status: 'error', duration: 5000 })
    } finally {
      setRunLoading(false)
    }
  }

  const handleUndo = async () => {
    try {
      const data = await undoPotteryEdit(sessionId)
      if (data.restored && data.results) setResults(data.results)
      else toast({ title: 'Nothing to undo', status: 'info' })
    } catch {
      toast({ title: 'Undo failed', status: 'error' })
    }
  }

  const handleExport = async () => {
    setExporting(true)
    try {
      const data = await exportSession(sessionId)
      setExported(true)
      toast({
        title: 'Exported',
        description: `Saved pottery labels to ${data.saved_finds} find(s) .ascota/classification.json`,
        status: 'success',
        duration: 4000,
      })
    } catch {
      toast({ title: 'Export failed', status: 'error' })
    } finally {
      setExporting(false)
    }
  }

  const handleQuickSwitch = async (itemId: string, label: (typeof POTTERY_LABELS)[number]) => {
    setSavingOverride(true)
    try {
      await updatePotteryResult(sessionId, itemId, label)
      setResults((prev) => ({
        ...prev,
        [itemId]: {
          ...prev[itemId],
          label,
          confidence: 1,
          p_pottery: prev[itemId]?.p_pottery,
        },
      }))
      toast({ title: 'Updated', status: 'success', duration: 1500 })
    } catch {
      toast({ title: 'Update failed', status: 'error' })
    } finally {
      setSavingOverride(false)
    }
  }

  const resolveSwipeForItem = useCallback(
    async (itemId: string, label: 'pottery' | 'non_pottery') => {
      if (!itemId) return
      setSavingOverride(true)
      try {
        await updatePotteryResult(sessionId, itemId, label)
        setResults((prev) => ({
          ...prev,
          [itemId]: {
            ...prev[itemId],
            label,
            confidence: 1,
            p_pottery: prev[itemId]?.p_pottery,
          },
        }))
      } catch {
        toast({ title: 'Update failed', status: 'error' })
      } finally {
        setSavingOverride(false)
      }
    },
    [sessionId, toast]
  )

  const resultsList: ResultEntry[] = useMemo(() => {
    return Object.entries(results).map(([item_id, r]) => {
      const [find_number, image_filename] = item_id.includes('_')
        ? item_id.split(/_(.*)/s).filter(Boolean)
        : ['', item_id]
      return {
        item_id,
        label: r.label,
        confidence: r.confidence,
        p_pottery: r.p_pottery,
        find_number: find_number || undefined,
        image_filename: image_filename || item_id,
      }
    })
  }, [results])

  const sortedByConfidence = useMemo(() => {
    const list = [...resultsList]
    list.sort((a, b) => (sortAsc ? a.confidence - b.confidence : b.confidence - a.confidence))
    return list
  }, [resultsList, sortAsc])

  const byCategory = useMemo(() => {
    const map: Record<string, ResultEntry[]> = {}
    for (const entry of sortedByConfidence) {
      const label = entry.label || 'unknown'
      if (!map[label]) map[label] = []
      map[label].push(entry)
    }
    return map
  }, [sortedByConfidence])

  const swipeQueue = useMemo(() => {
    return sortedByConfidence.filter((e) => e.confidence < swipeThreshold)
  }, [sortedByConfidence, swipeThreshold])

  const swipeCurrent = swipeQueue[0]

  useEffect(() => {
    if (viewMode !== 'swipe' || !hasRun) return
    swipeFocusRef.current?.focus()
  }, [viewMode, hasRun, swipeCurrent?.item_id])

  useEffect(() => {
    if (viewMode !== 'swipe' || !hasRun) return
    const id = swipeCurrent?.item_id
    if (!id) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        void resolveSwipeForItem(id, 'non_pottery')
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        void resolveSwipeForItem(id, 'pottery')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [viewMode, hasRun, swipeCurrent?.item_id, resolveSwipeForItem])

  if (exported) {
    return (
      <Container maxW="container.lg" py={10}>
        <VStack spacing={6}>
          <Heading size="lg">Export complete</Heading>
          <Text color="gray.600">Pottery labels were saved to each find&apos;s .ascota/classification.json under key &quot;pottery&quot;.</Text>
          <Button colorScheme="brand" onClick={onReset}>
            Start over
          </Button>
        </VStack>
      </Container>
    )
  }

  return (
    <Container maxW={!hasRun ? 'container.lg' : 'container.xl'} py={!hasRun ? 10 : 6}>
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between" flexWrap="wrap">
          <Heading size="lg">Pottery vs non-pottery</Heading>
          <Button size="sm" variant="outline" onClick={onReset}>
            Back to setup
          </Button>
        </HStack>

        {!hasRun ? (
          <VStack spacing={8} align="stretch" textAlign="center">
            <Box>
              <VStack align="stretch" spacing={4} maxW="md" mx="auto">
                <Text fontSize="sm" color="gray.600">
                  DINOv2-large + sklearn binary model. Export when done so Type / Decoration / Color sessions can run on pottery images only.
                </Text>
                <FormControl>
                  <FormLabel>Resolution</FormLabel>
                  <Select
                    value={options.resolution}
                    onChange={(e) =>
                      setOptions((o) => ({
                        ...o,
                        resolution: Number(e.target.value) as 1000 | 1500 | 3000,
                      }))
                    }
                  >
                    <option value={1000}>1000px</option>
                    <option value={1500}>1500px</option>
                    <option value={3000}>3000px</option>
                  </Select>
                </FormControl>
                <Button
                  colorScheme="brand"
                  size="lg"
                  onClick={handleRun}
                  isLoading={runLoading}
                  loadingText="Classifying..."
                >
                  Run classification
                </Button>
              </VStack>
            </Box>

            {runLoading && (
              <Flex
                position="fixed"
                inset={0}
                bg="blackAlpha.600"
                align="center"
                justify="center"
                zIndex={10}
              >
                <VStack spacing={4} bg="white" p={8} borderRadius="lg">
                  <Spinner size="xl" />
                  <Text fontWeight="medium">Classifying images...</Text>
                </VStack>
              </Flex>
            )}
          </VStack>
        ) : (
          <>
            <HStack spacing={4} flexWrap="wrap" align="center">
              <Button size="sm" onClick={handleUndo}>
                Undo
              </Button>
              <Button
                colorScheme="green"
                size="sm"
                onClick={handleExport}
                isLoading={exporting}
                loadingText="Exporting..."
              >
                Export
              </Button>
              <HStack spacing={2}>
                <Button
                  size="sm"
                  variant={viewMode === 'review' ? 'solid' : 'outline'}
                  colorScheme="brand"
                  onClick={() => setViewMode('review')}
                >
                  Review
                </Button>
                <Button
                  size="sm"
                  variant={viewMode === 'swipe' ? 'solid' : 'outline'}
                  colorScheme="brand"
                  onClick={() => setViewMode('swipe')}
                >
                  Quick swipe
                </Button>
              </HStack>
              <Select
                w="auto"
                size="sm"
                value={sortAsc ? 'asc' : 'desc'}
                onChange={(e) => setSortAsc(e.target.value === 'asc')}
              >
                <option value="asc">Confidence: low to high</option>
                <option value="desc">Confidence: high to low</option>
              </Select>
              <Tooltip label="Show image with transparent background (mask applied)">
                <HStack spacing={2} whiteSpace="nowrap">
                  <Text fontSize="sm">Transparent mode</Text>
                  <Switch
                    size="sm"
                    isChecked={transparentMode}
                    onChange={(e) => setTransparentMode(e.target.checked)}
                  />
                </HStack>
              </Tooltip>
              <Tooltip label="Crop to object only (zoom to object); uses masked image">
                <HStack spacing={2} whiteSpace="nowrap">
                  <Text fontSize="sm">Focus on object</Text>
                  <Switch
                    size="sm"
                    isChecked={focusOnPottery}
                    onChange={(e) => setFocusOnPottery(e.target.checked)}
                  />
                </HStack>
              </Tooltip>
            </HStack>

            {viewMode === 'swipe' && (
              <Box borderWidth="1px" borderRadius="md" p={4} bg="gray.50">
                <VStack align="stretch" spacing={4}>
                  <Text fontSize="sm" fontWeight="medium">
                    Confidence threshold: {(swipeThreshold * 100).toFixed(0)}%
                  </Text>
                  <Text fontSize="xs" color="gray.600">
                    Images with model confidence below this value appear in the queue. Arrow Left → non-pottery, Arrow Right → pottery (or use buttons).
                  </Text>
                  <Slider
                    value={swipeThreshold}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={setSwipeThreshold}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                  <Text fontSize="sm" color="gray.600">
                    Queue: {swipeQueue.length} image(s) below threshold
                  </Text>
                </VStack>
              </Box>
            )}

            <Divider />

            {viewMode === 'swipe' ? (
              <Box
                ref={swipeFocusRef}
                tabIndex={0}
                outline="none"
                borderRadius="md"
                _focus={{ boxShadow: 'outline' }}
              >
                {!swipeCurrent ? (
                  <Text color="gray.600" textAlign="center" py={8}>
                    No images below the threshold. Raise the threshold or switch to Review.
                  </Text>
                ) : (
                  <VStack spacing={4} align="stretch" maxW="lg" mx="auto">
                    <Text fontSize="sm" textAlign="center" color="gray.600">
                      {swipeCurrent.find_number}_{swipeCurrent.image_filename}
                    </Text>
                    <Box
                      h="320px"
                      bg="gray.200"
                      borderRadius="lg"
                      borderWidth="1px"
                      backgroundImage={`url(${getImageUrl(sessionId, swipeCurrent.item_id, {
                        transparentMode,
                        focusOnPottery,
                      })})`}
                      backgroundSize="contain"
                      backgroundRepeat="no-repeat"
                      backgroundPosition="center"
                    />
                    <Text fontSize="sm" textAlign="center">
                      Confidence:{' '}
                      <Text as="span" fontWeight="bold">
                        {(swipeCurrent.confidence * 100).toFixed(1)}% {potteryLabelDisplay(swipeCurrent.label)}
                      </Text>
                    </Text>
                    <HStack justify="center" spacing={4}>
                      <Button
                        colorScheme="gray"
                        size="lg"
                        onClick={() => void resolveSwipeForItem(swipeCurrent.item_id, 'non_pottery')}
                        isDisabled={savingOverride}
                      >
                        ← Non-pottery
                      </Button>
                      <Button
                        colorScheme="brand"
                        size="lg"
                        onClick={() => void resolveSwipeForItem(swipeCurrent.item_id, 'pottery')}
                        isDisabled={savingOverride}
                      >
                        Pottery →
                      </Button>
                    </HStack>
                  </VStack>
                )}
              </Box>
            ) : (
              POTTERY_LABELS.map((label) => {
                const items = byCategory[label] || []
                if (items.length === 0) return null
                return (
                  <Box key={label}>
                    <Heading size="sm" mb={3} color="gray.700">
                      {label} ({items.length})
                    </Heading>
                    <SimpleGrid columns={{ base: 2, md: 4, lg: 6 }} spacing={3}>
                      {items.map((entry) => (
                        <Menu key={entry.item_id} placement="right-start" isLazy>
                          <MenuButton
                            as={Box}
                            textAlign="left"
                            borderWidth="1px"
                            borderRadius="md"
                            overflow="hidden"
                            cursor="pointer"
                            _hover={{ borderColor: 'brand.500', shadow: 'md' }}
                            transition="all 0.15s"
                          >
                            <Box
                              h="120px"
                              bg={transparentMode || focusOnPottery ? 'gray.200' : 'gray.100'}
                              backgroundImage={`url(${getImageUrl(sessionId, entry.item_id, {
                                transparentMode,
                                focusOnPottery,
                              })})`}
                              backgroundSize={transparentMode || focusOnPottery ? 'contain' : 'cover'}
                              backgroundRepeat="no-repeat"
                              backgroundPosition="center"
                            />
                            <Box p={2}>
                              <Text fontSize="xs" noOfLines={1}>
                                {entry.label}
                              </Text>
                              <Text fontSize="xs" color="gray.500">
                                {(entry.confidence * 100).toFixed(0)}%
                              </Text>
                            </Box>
                          </MenuButton>
                          <MenuList minW="160px" py={0}>
                            {POTTERY_LABELS.map((opt) => (
                              <MenuItem
                                key={opt}
                                onClick={() => void handleQuickSwitch(entry.item_id, opt)}
                                fontWeight={entry.label === opt ? 'semibold' : 'normal'}
                                bg={entry.label === opt ? 'brand.50' : undefined}
                                isDisabled={savingOverride}
                              >
                                {opt}
                              </MenuItem>
                            ))}
                          </MenuList>
                        </Menu>
                      ))}
                    </SimpleGrid>
                  </Box>
                )
              })
            )}
          </>
        )}
      </VStack>
    </Container>
  )
}

export default PotteryClassification
