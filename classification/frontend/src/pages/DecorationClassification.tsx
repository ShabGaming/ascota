import { useState, useEffect, useMemo } from 'react'
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
  Divider,
  Input,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  useDisclosure,
  Switch,
  Tooltip,
} from '@chakra-ui/react'
import {
  runDecorationClassification,
  getDecorationResults,
  getSession,
  updateDecorationResult,
  undoDecorationEdit,
  exportSession,
  getImageUrl,
  DecorationRunOptions,
  TypeResultItem,
} from '../api/client'

const DECORATION_LABELS = ['Impressed', 'Incised'] as const

interface DecorationClassificationProps {
  sessionId: string
  onReset: () => void
}

interface ResultEntry extends TypeResultItem {
  find_number?: string
  image_filename?: string
}

function DecorationClassification({ sessionId, onReset }: DecorationClassificationProps) {
  const [options, setOptions] = useState<DecorationRunOptions>({ resolution: 1500 })
  const [results, setResults] = useState<Record<string, { label: string; confidence: number }>>({})
  const [runLoading, setRunLoading] = useState(false)
  const [hasRun, setHasRun] = useState(false)
  const [sortAsc, setSortAsc] = useState(true)
  const [transparentMode, setTransparentMode] = useState(false)
  const [focusOnPottery, setFocusOnPottery] = useState(false)
  const [savingOverride, setSavingOverride] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState(false)
  const [customForItemId, setCustomForItemId] = useState<string | null>(null)
  const [customLabel, setCustomLabel] = useState('')
  const { isOpen: isCustomOpen, onOpen: onCustomOpen, onClose: onCustomClose } = useDisclosure()
  const toast = useToast()

  const fetchResults = async () => {
    try {
      const data = await getDecorationResults(sessionId)
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
          return getDecorationResults(sessionId).then((data) => {
            if (data?.results) setResults(data.results)
          })
        }
      })
      .catch(() => {})
  }, [sessionId])

  const handleRun = async () => {
    setRunLoading(true)
    try {
      await runDecorationClassification(sessionId, options)
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
      const data = await undoDecorationEdit(sessionId)
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
        description: `Saved to ${data.saved_finds} find(s) .ascota/classification.json`,
        status: 'success',
        duration: 4000,
      })
    } catch {
      toast({ title: 'Export failed', status: 'error' })
    } finally {
      setExporting(false)
    }
  }

  const handleQuickSwitch = async (itemId: string, label: string) => {
    setSavingOverride(true)
    try {
      await updateDecorationResult(sessionId, itemId, label)
      setResults((prev) => ({ ...prev, [itemId]: { label, confidence: 1 } }))
      toast({ title: 'Classification updated', status: 'success', duration: 1500 })
    } catch {
      toast({ title: 'Update failed', status: 'error' })
    } finally {
      setSavingOverride(false)
    }
  }

  const handleOpenCustom = (itemId: string) => {
    setCustomForItemId(itemId)
    setCustomLabel('')
    onCustomOpen()
  }

  const handleCustomSave = async () => {
    const label = customLabel.trim()
    if (!customForItemId || !label) return
    onCustomClose()
    const itemId = customForItemId
    setCustomForItemId(null)
    setCustomLabel('')
    await handleQuickSwitch(itemId, label)
  }

  const resultsList: ResultEntry[] = useMemo(() => {
    return Object.entries(results).map(([item_id, r]) => {
      const parts = item_id.includes('_') ? item_id.split(/_(.*)/s).filter(Boolean) : ['', item_id]
      return {
        item_id,
        label: r.label,
        confidence: r.confidence,
        find_number: parts[0] || undefined,
        image_filename: parts[1] || item_id,
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
      const label = entry.label || 'Unknown'
      if (!map[label]) map[label] = []
      map[label].push(entry)
    }
    return map
  }, [sortedByConfidence])

  const categoryOrder = useMemo(() => {
    const labels = new Set<string>()
    DECORATION_LABELS.forEach((l) => labels.add(l))
    sortedByConfidence.forEach((e) => labels.add(e.label || 'Unknown'))
    return Array.from(labels)
  }, [sortedByConfidence])

  /** Custom labels used in this session (for dropdown options). */
  const sessionCustomLabels = useMemo(() => {
    const builtIn = new Set(DECORATION_LABELS)
    const custom = new Set<string>()
    Object.values(results).forEach((r) => {
      const label = (r.label || '').trim()
      if (label && !builtIn.has(label)) custom.add(label)
    })
    return Array.from(custom).sort()
  }, [results])

  if (exported) {
    return (
      <Container maxW="container.lg" py={10}>
        <VStack spacing={6}>
          <Heading size="lg">Export complete</Heading>
          <Text color="gray.600">Results were saved to each find&apos;s .ascota/classification.json.</Text>
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
          <Heading size="lg">Decoration / Pattern classification</Heading>
          <Button size="sm" variant="outline" onClick={onReset}>
            Back to setup
          </Button>
        </HStack>

        {!hasRun ? (
          <VStack spacing={8} align="stretch" textAlign="center">
            <Box>
              <VStack align="stretch" spacing={4} maxW="md" mx="auto">
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
              <Tooltip label="Crop to pottery only (zoom to object); uses masked image">
                <HStack spacing={2} whiteSpace="nowrap">
                  <Text fontSize="sm">Focus on pottery</Text>
                  <Switch
                    size="sm"
                    isChecked={focusOnPottery}
                    onChange={(e) => setFocusOnPottery(e.target.checked)}
                  />
                </HStack>
              </Tooltip>
            </HStack>
            <Divider />
            {categoryOrder.map((label) => {
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
                        <MenuList minW="180px" py={0}>
                          {DECORATION_LABELS.map((lbl) => (
                            <MenuItem
                              key={lbl}
                              onClick={() => handleQuickSwitch(entry.item_id, lbl)}
                              fontWeight={entry.label === lbl ? 'semibold' : 'normal'}
                              bg={entry.label === lbl ? 'brand.50' : undefined}
                              isDisabled={savingOverride}
                            >
                              {lbl}
                            </MenuItem>
                          ))}
                          {sessionCustomLabels.map((lbl) => (
                            <MenuItem
                              key={lbl}
                              onClick={() => handleQuickSwitch(entry.item_id, lbl)}
                              fontWeight={entry.label === lbl ? 'semibold' : 'normal'}
                              bg={entry.label === lbl ? 'brand.50' : undefined}
                              isDisabled={savingOverride}
                            >
                              {lbl}
                            </MenuItem>
                          ))}
                          <MenuItem
                            onClick={() => handleOpenCustom(entry.item_id)}
                            isDisabled={savingOverride}
                          >
                            Other (custom pattern)…
                          </MenuItem>
                        </MenuList>
                      </Menu>
                    ))}
                  </SimpleGrid>
                </Box>
              )
            })}
          </>
        )}
      </VStack>

      <Modal isOpen={isCustomOpen} onClose={onCustomClose} size="sm">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Custom pattern type</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Input
              placeholder="e.g. Painted, Stamped"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCustomSave()}
            />
          </ModalBody>
          <ModalFooter>
            <Button size="sm" variant="ghost" mr={2} onClick={onCustomClose}>
              Cancel
            </Button>
            <Button size="sm" colorScheme="brand" onClick={handleCustomSave}>
              Save
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Container>
  )
}

export default DecorationClassification
