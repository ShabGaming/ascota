import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Box,
  Button,
  Container,
  Heading,
  VStack,
  HStack,
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
  Switch,
  Tooltip,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  FormControl,
  FormLabel,
  Select,
  Wrap,
  WrapItem,
} from '@chakra-ui/react'
import {
  runColorClustering,
  getColorResults,
  getSession,
  reclusterColor,
  updateColorResult,
  setColorClusterName,
  undoColorEdit,
  exportSession,
  getImageUrl,
  ColorResultsResponse,
  ReclusterOptions,
} from '../api/client'

const DEFAULT_MIN_CLUSTER_SIZE = 5
const DEFAULT_MIN_SAMPLES = 5
const DEFAULT_CLUSTER_SELECTION_EPSILON = 0
const DEFAULT_CLUSTER_SELECTION_METHOD = 'eom'
const RESOLUTION_OPTIONS = [1000, 1500, 3000] as const

interface ColorClassificationProps {
  sessionId: string
  onReset: () => void
}

function ColorClassification({ sessionId, onReset }: ColorClassificationProps) {
  const [results, setResults] = useState<Record<string, { cluster_id: number }>>({})
  const [clusterNames, setClusterNames] = useState<Record<string, string>>({})
  const [clusters, setClusters] = useState<Array<{ cluster_id: number; item_ids: string[] }>>([])
  const [noiseItemIds, setNoiseItemIds] = useState<string[]>([])
  const [hasRun, setHasRun] = useState(false)
  const [runLoading, setRunLoading] = useState(false)
  const [reclusterLoading, setReclusterLoading] = useState(false)
  const [transparentMode, setTransparentMode] = useState(false)
  const [focusOnPottery, setFocusOnPottery] = useState(false)
  const [savingOverride, setSavingOverride] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState(false)
  const [minClusterSize, setMinClusterSize] = useState(DEFAULT_MIN_CLUSTER_SIZE)
  const [minSamples, setMinSamples] = useState(DEFAULT_MIN_SAMPLES)
  const [clusterSelectionEpsilon, setClusterSelectionEpsilon] = useState(DEFAULT_CLUSTER_SELECTION_EPSILON)
  const [clusterSelectionMethod, setClusterSelectionMethod] = useState(DEFAULT_CLUSTER_SELECTION_METHOD)
  const [resolution, setResolution] = useState<1000 | 1500 | 3000>(1500)
  const toast = useToast()

  const applyResults = useCallback((data: ColorResultsResponse) => {
    setResults(data.results || {})
    setClusterNames(data.cluster_names || {})
    setClusters(data.clusters || [])
    setNoiseItemIds(data.noise_item_ids || [])
  }, [])

  const fetchResults = useCallback(async () => {
    try {
      const data = await getColorResults(sessionId)
      applyResults(data)
    } catch {
      toast({ title: 'Failed to load results', status: 'error' })
    }
  }, [sessionId, applyResults, toast])

  useEffect(() => {
    if (hasRun && sessionId) fetchResults()
  }, [sessionId, hasRun, fetchResults])

  useEffect(() => {
    if (!sessionId) return
    getSession(sessionId)
      .then((session) => {
        if (session.has_results) {
          setHasRun(true)
          return getColorResults(sessionId).then((data) => {
            if (data?.results) applyResults(data)
          })
        }
      })
      .catch(() => {})
  }, [sessionId, applyResults])

  const handleRun = async () => {
    setRunLoading(true)
    try {
      const data = await runColorClustering(sessionId, { resolution })
      setHasRun(true)
      applyResults(data)
      toast({ title: 'Clustering complete', status: 'success', duration: 2000 })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Run failed'
      toast({ title: 'Clustering failed', description: msg, status: 'error', duration: 5000 })
    } finally {
      setRunLoading(false)
    }
  }

  const handleRecluster = async () => {
    setReclusterLoading(true)
    try {
      const options: ReclusterOptions = {
        min_cluster_size: minClusterSize,
        min_samples: minSamples,
        cluster_selection_epsilon: clusterSelectionEpsilon,
        cluster_selection_method: clusterSelectionMethod,
      }
      const data = await reclusterColor(sessionId, options)
      applyResults(data)
      toast({ title: 'Reclustered', status: 'success', duration: 2000 })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Recluster failed'
      toast({ title: 'Recluster failed', description: msg, status: 'error', duration: 5000 })
    } finally {
      setReclusterLoading(false)
    }
  }

  const handleUndo = async () => {
    try {
      const data = await undoColorEdit(sessionId)
      if (data.restored && data.results) {
        setResults(data.results)
        await fetchResults()
      } else {
        toast({ title: 'Nothing to undo', status: 'info' })
      }
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

  const handleMove = async (itemId: string, clusterId: number) => {
    setSavingOverride(true)
    try {
      await updateColorResult(sessionId, itemId, clusterId)
      await fetchResults()
      toast({ title: 'Moved', status: 'success', duration: 1500 })
    } catch {
      toast({ title: 'Move failed', status: 'error' })
    } finally {
      setSavingOverride(false)
    }
  }

  const handleRenameCluster = async (clusterId: number, name: string) => {
    const trimmed = name.trim()
    if (!trimmed) return
    try {
      await setColorClusterName(sessionId, clusterId, trimmed)
      setClusterNames((prev) => ({ ...prev, [String(clusterId)]: trimmed }))
    } catch {
      toast({ title: 'Rename failed', status: 'error' })
    }
  }

  const clusterDisplayName = (clusterId: number) =>
    clusterNames[String(clusterId)] ?? (clusterId === -1 ? 'Noise' : `Cluster ${clusterId}`)

  /** Next unused cluster id for "Create new cluster". */
  const nextNewClusterId = useMemo(() => {
    const ids = clusters.map((c) => c.cluster_id).filter((id) => id >= 0)
    return ids.length ? Math.max(...ids) + 1 : 0
  }, [clusters])

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
          <Heading size="lg">Color clustering</Heading>
          <Button size="sm" variant="outline" onClick={onReset}>
            Back to setup
          </Button>
        </HStack>

        {!hasRun ? (
          <VStack spacing={8} align="stretch" textAlign="center">
            <Box>
              <VStack align="stretch" spacing={4} maxW="md" mx="auto">
                <FormControl>
                  <FormLabel>Preview resolution</FormLabel>
                  <Select
                    value={resolution}
                    onChange={(e) =>
                      setResolution(Number(e.target.value) as 1000 | 1500 | 3000)
                    }
                  >
                    {RESOLUTION_OPTIONS.map((r) => (
                      <option key={r} value={r}>
                        {r}px
                      </option>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  colorScheme="brand"
                  size="lg"
                  onClick={handleRun}
                  isLoading={runLoading}
                  loadingText="Extracting features and clustering..."
                >
                  Run color clustering
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
                  <Text fontWeight="medium">Extracting features and clustering...</Text>
                </VStack>
              </Flex>
            )}
          </VStack>
        ) : (
          <>
            <Wrap spacing={4} align="center">
              <WrapItem>
                <Tooltip label="Minimum size of a cluster; smaller groups become noise.">
                  <FormControl w="100px">
                    <FormLabel fontSize="xs" mb={1}>
                      min_cluster_size
                    </FormLabel>
                    <NumberInput
                      size="sm"
                      value={minClusterSize}
                      min={2}
                      max={50}
                      onChange={(_s, n) => setMinClusterSize(n || 2)}
                      w="100px"
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                </Tooltip>
              </WrapItem>
              <WrapItem>
                <Tooltip label="Core point neighborhood; often same as min_cluster_size.">
                  <FormControl w="90px">
                    <FormLabel fontSize="xs" mb={1}>
                      min_samples
                    </FormLabel>
                    <NumberInput
                      size="sm"
                      value={minSamples}
                      min={1}
                      max={50}
                      onChange={(_s, n) => setMinSamples(n || 1)}
                      w="90px"
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                </Tooltip>
              </WrapItem>
              <WrapItem>
                <Tooltip label="Distance threshold for merging clusters (0 = no merge by epsilon).">
                  <FormControl w="80px">
                    <FormLabel fontSize="xs" mb={1}>
                      epsilon
                    </FormLabel>
                    <NumberInput
                      size="sm"
                      value={clusterSelectionEpsilon}
                      min={0}
                      max={2}
                      step={0.1}
                      onChange={(_s, n) => setClusterSelectionEpsilon(n ?? 0)}
                      w="80px"
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                </Tooltip>
              </WrapItem>
              <WrapItem>
                <Tooltip label="eom = excess of mass; leaf = leaf extraction.">
                  <FormControl w="100px">
                    <FormLabel fontSize="xs" mb={1}>
                      method
                    </FormLabel>
                    <Select
                      size="sm"
                      value={clusterSelectionMethod}
                      onChange={(e) => setClusterSelectionMethod(e.target.value)}
                      w="100px"
                    >
                      <option value="eom">eom</option>
                      <option value="leaf">leaf</option>
                    </Select>
                  </FormControl>
                </Tooltip>
              </WrapItem>
              <WrapItem>
                <Button size="sm" onClick={handleRecluster} isLoading={reclusterLoading} loadingText="Reclustering...">
                  Recluster
                </Button>
              </WrapItem>
              <WrapItem>
                <Button size="sm" onClick={handleUndo}>
                  Undo
                </Button>
              </WrapItem>
              <WrapItem>
                <Button
                  colorScheme="green"
                  size="sm"
                  onClick={handleExport}
                  isLoading={exporting}
                  loadingText="Exporting..."
                >
                  Export
                </Button>
              </WrapItem>
              <WrapItem>
                <Tooltip label="Show image with transparent background (mask applied)">
                  <HStack spacing={2} whiteSpace="nowrap">
                    <Text fontSize="sm">Transparent</Text>
                    <Switch
                      size="sm"
                      isChecked={transparentMode}
                      onChange={(e) => setTransparentMode(e.target.checked)}
                    />
                  </HStack>
                </Tooltip>
              </WrapItem>
              <WrapItem>
                <Tooltip label="Crop to pottery only (zoom to object); uses masked image">
                  <HStack spacing={2} whiteSpace="nowrap">
                    <Text fontSize="sm">Focus</Text>
                    <Switch
                      size="sm"
                      isChecked={focusOnPottery}
                      onChange={(e) => setFocusOnPottery(e.target.checked)}
                    />
                  </HStack>
                </Tooltip>
              </WrapItem>
            </Wrap>
            <Divider />
            {clusters.map((cluster) => (
              <Box key={cluster.cluster_id}>
                <HStack mb={3} spacing={3} align="center">
                  <Input
                    size="sm"
                    maxW="200px"
                    value={clusterNames[String(cluster.cluster_id)] ?? `Cluster ${cluster.cluster_id}`}
                    onChange={(e) =>
                      setClusterNames((prev) => ({ ...prev, [String(cluster.cluster_id)]: e.target.value }))
                    }
                    onBlur={(e) => handleRenameCluster(cluster.cluster_id, e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                    placeholder={`Cluster ${cluster.cluster_id}`}
                  />
                  <Text fontSize="sm" color="gray.600">
                    ({cluster.item_ids.length})
                  </Text>
                </HStack>
                <SimpleGrid columns={{ base: 2, md: 4, lg: 6 }} spacing={3}>
                  {cluster.item_ids.map((itemId) => (
                    <Menu key={itemId} placement="right-start" isLazy>
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
                          backgroundImage={`url(${getImageUrl(sessionId, itemId, {
                            transparentMode,
                            focusOnPottery,
                          })})`}
                          backgroundSize={transparentMode || focusOnPottery ? 'contain' : 'cover'}
                          backgroundRepeat="no-repeat"
                          backgroundPosition="center"
                        />
                        <Box p={2} bg="white">
                          <Text fontSize="xs" noOfLines={1}>
                            {clusterDisplayName(results[itemId]?.cluster_id ?? cluster.cluster_id)}
                          </Text>
                          <Text fontSize="xs" color="gray.500">
                            Click to move
                          </Text>
                        </Box>
                      </MenuButton>
                      <MenuList minW="180px" py={0}>
                        {clusters
                          .filter((c) => c.cluster_id !== cluster.cluster_id)
                          .map((c) => (
                            <MenuItem
                              key={c.cluster_id}
                              onClick={() => handleMove(itemId, c.cluster_id)}
                              isDisabled={savingOverride}
                            >
                              Move to {clusterDisplayName(c.cluster_id)}
                            </MenuItem>
                          ))}
                        <MenuItem onClick={() => handleMove(itemId, -1)} isDisabled={savingOverride}>
                          Move to Noise
                        </MenuItem>
                        <MenuItem
                          onClick={() => handleMove(itemId, nextNewClusterId)}
                          isDisabled={savingOverride}
                        >
                          Create new cluster
                        </MenuItem>
                      </MenuList>
                    </Menu>
                  ))}
                </SimpleGrid>
              </Box>
            ))}
            {noiseItemIds.length > 0 && (
              <Box>
                <HStack mb={3} spacing={3} align="center">
                  <Input
                    size="sm"
                    maxW="200px"
                    value={clusterNames['-1'] ?? 'Noise'}
                    onChange={(e) => setClusterNames((prev) => ({ ...prev, '-1': e.target.value }))}
                    onBlur={(e) => handleRenameCluster(-1, e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                    placeholder="Noise"
                  />
                  <Text fontSize="sm" color="gray.600">
                    ({noiseItemIds.length})
                  </Text>
                </HStack>
                <SimpleGrid columns={{ base: 2, md: 4, lg: 6 }} spacing={3}>
                  {noiseItemIds.map((itemId) => (
                    <Menu key={itemId} placement="right-start" isLazy>
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
                          backgroundImage={`url(${getImageUrl(sessionId, itemId, {
                            transparentMode,
                            focusOnPottery,
                          })})`}
                          backgroundSize={transparentMode || focusOnPottery ? 'contain' : 'cover'}
                          backgroundRepeat="no-repeat"
                          backgroundPosition="center"
                        />
                        <Box p={2} bg="white">
                          <Text fontSize="xs" noOfLines={1}>
                            {clusterDisplayName(results[itemId]?.cluster_id ?? -1)}
                          </Text>
                          <Text fontSize="xs" color="gray.500">
                            Click to move
                          </Text>
                        </Box>
                      </MenuButton>
                      <MenuList minW="180px" py={0}>
                        {clusters.map((c) => (
                            <MenuItem
                              key={c.cluster_id}
                              onClick={() => handleMove(itemId, c.cluster_id)}
                              isDisabled={savingOverride}
                            >
                              Move to {clusterDisplayName(c.cluster_id)}
                            </MenuItem>
                          ))}
                        <MenuItem
                          onClick={() => handleMove(itemId, nextNewClusterId)}
                          isDisabled={savingOverride}
                        >
                          Create new cluster
                        </MenuItem>
                      </MenuList>
                    </Menu>
                  ))}
                </SimpleGrid>
              </Box>
            )}
          </>
        )}
      </VStack>
    </Container>
  )
}

export default ColorClassification
