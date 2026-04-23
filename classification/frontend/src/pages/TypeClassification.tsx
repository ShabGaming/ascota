import { useState, useEffect, useMemo } from 'react'
import {
  Box,
  Button,
  Container,
  Heading,
  VStack,
  HStack,
  Checkbox,
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
} from '@chakra-ui/react'
import {
  runTypeClassification,
  getTypeResults,
  getSession,
  updateTypeResult,
  undoTypeEdit,
  exportSession,
  getImageUrl,
  TypeRunOptions,
  TypeResultItem,
} from '../api/client'

const TYPE_LABELS = [
  'body',
  'base',
  'rim',
  'appendage',
  'lid',
  'rim-handle',
  'spout',
  'rounded',
  'body-decorated',
  'tile',
] as const

const APPENDAGE_TOOLTIP =
  'If our models classify an object as an appendage, we use additional models to classify the appendage subtype (e.g. lid, spout). May be less accurate than the main type model.'

interface TypeClassificationProps {
  sessionId: string
  onReset: () => void
}

interface ResultEntry extends TypeResultItem {
  find_number?: string
  image_filename?: string
}

function TypeClassification({ sessionId, onReset }: TypeClassificationProps) {
  const [options, setOptions] = useState<TypeRunOptions>({
    enable_appendage_subtype: true,
    resolution: 1500,
  })
  const [results, setResults] = useState<Record<string, { label: string; confidence: number }>>({})
  const [runLoading, setRunLoading] = useState(false)
  const [hasRun, setHasRun] = useState(false)
  const [sortAsc, setSortAsc] = useState(true)
  const [transparentMode, setTransparentMode] = useState(false)
  const [focusOnPottery, setFocusOnPottery] = useState(false)
  const [savingOverride, setSavingOverride] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState(false)
  const toast = useToast()

  const fetchResults = async () => {
    try {
      const data = await getTypeResults(sessionId)
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
          return getTypeResults(sessionId).then((data) => {
            if (data?.results) setResults(data.results)
          })
        }
      })
      .catch(() => {})
  }, [sessionId])

  const handleRun = async () => {
    setRunLoading(true)
    try {
      await runTypeClassification(sessionId, options)
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
      const data = await undoTypeEdit(sessionId)
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
      await updateTypeResult(sessionId, itemId, label)
      setResults((prev) => ({
        ...prev,
        [itemId]: { label, confidence: 1 },
      }))
      toast({ title: 'Classification updated', status: 'success', duration: 1500 })
    } catch {
      toast({ title: 'Update failed', status: 'error' })
    } finally {
      setSavingOverride(false)
    }
  }

  const resultsList: ResultEntry[] = useMemo(() => {
    return Object.entries(results).map(([item_id, r]) => {
      const [find_number, image_filename] = item_id.includes('_')
        ? item_id.split(/_(.*)/s).filter(Boolean)
        : ['', item_id]
      return {
        item_id,
        label: r.label,
        confidence: r.confidence,
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
          <Heading size="lg">Type classification</Heading>
          <Button size="sm" variant="outline" onClick={onReset}>
            Back to setup
          </Button>
        </HStack>

        {!hasRun ? (
          <VStack spacing={8} align="stretch" textAlign="center">
            <Box>
              <VStack align="stretch" spacing={4} maxW="md" mx="auto">
                <FormControl>
                  <Tooltip label={APPENDAGE_TOOLTIP} hasArrow placement="top">
                    <Box>
                      <Checkbox
                        isChecked={options.enable_appendage_subtype}
                        onChange={(e) =>
                          setOptions((o) => ({ ...o, enable_appendage_subtype: e.target.checked }))
                        }
                      >
                        Enable Appendage subtype classification
                      </Checkbox>
                    </Box>
                  </Tooltip>
                </FormControl>
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

            {TYPE_LABELS.map((label) => {
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
                          {TYPE_LABELS.map((label) => (
                            <MenuItem
                              key={label}
                              onClick={() => handleQuickSwitch(entry.item_id, label)}
                              fontWeight={entry.label === label ? 'semibold' : 'normal'}
                              bg={entry.label === label ? 'brand.50' : undefined}
                              isDisabled={savingOverride}
                            >
                              {label}
                            </MenuItem>
                          ))}
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
    </Container>
  )
}

export default TypeClassification
