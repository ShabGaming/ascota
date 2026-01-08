import { useState, useRef } from 'react'
import {
  Box,
  Container,
  HStack,
  Button,
  Text,
  Progress,
  VStack,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  List,
  ListItem,
  useDisclosure,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from '@chakra-ui/react'
import { useMutation, useQuery } from 'react-query'
import { startExport, getJobStatus, ExportSummary, deleteSession } from '../api/client'

interface ExportBarProps {
  sessionId: string
  onExportComplete?: () => void
  isEditing?: boolean
}

function ExportBar({ sessionId, onExportComplete, isEditing = false }: ExportBarProps) {
  const [exportJobId, setExportJobId] = useState<string | null>(null)
  const [exportSummary, setExportSummary] = useState<ExportSummary | null>(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const { isOpen: isConfirmOpen, onOpen: onConfirmOpen, onClose: onConfirmClose } = useDisclosure()
  const cancelRef = useRef<HTMLButtonElement>(null)
  const toast = useToast()
  
  const exportMutation = useMutation(
    () => startExport(sessionId),
    {
      onSuccess: (data) => {
        setExportJobId(data.job_id)
        toast({
          title: 'Export started',
          status: 'info',
          duration: 2000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Failed to start export',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  const { data: jobStatus } = useQuery(
    ['exportJob', sessionId, exportJobId],
    () => getJobStatus(sessionId, exportJobId!),
    {
      enabled: !!exportJobId,
      refetchInterval: (data) => {
        if (!data || data.status === 'completed' || data.status === 'failed') {
          return false
        }
        return 1000
      },
      onSuccess: async (data) => {
        if (data.status === 'completed') {
          setExportSummary(data.result as ExportSummary)
          onOpen()
          setExportJobId(null)
          
          toast({
            title: 'Export completed',
            description: `${data.result.total_files_written} files written`,
            status: 'success',
            duration: 5000,
          })
          
          // Delete the session after successful export
          try {
            await deleteSession(sessionId)
            toast({
              title: 'Session deleted',
              description: 'Session has been cleaned up',
              status: 'info',
              duration: 2000,
            })
            // Call the callback to reset/navigate
            if (onExportComplete) {
              onExportComplete()
            }
          } catch (error) {
            console.error('Failed to delete session:', error)
            toast({
              title: 'Failed to delete session',
              description: error instanceof Error ? error.message : 'Unknown error',
              status: 'warning',
              duration: 3000,
            })
          }
        } else if (data.status === 'failed') {
          setExportJobId(null)
          toast({
            title: 'Export failed',
            description: data.error,
            status: 'error',
            duration: 5000,
          })
        }
      },
    }
  )
  
  const isExporting = !!exportJobId
  const progress = jobStatus?.progress || 0
  
  const handleExportClick = () => {
    onConfirmOpen()
  }
  
  const handleConfirmExport = () => {
    onConfirmClose()
    exportMutation.mutate()
  }
  
  // Hide export bar when editing
  if (isEditing && !isExporting) {
    return null
  }
  
  return (
    <>
      <Box
        position="fixed"
        bottom={0}
        left={0}
        right={0}
        bg="white"
        borderTop="1px"
        borderColor="gray.200"
        py={3}
        zIndex={10}
        boxShadow="lg"
      >
        <Container maxW="full" px={6}>
          {isExporting ? (
            <VStack spacing={2}>
              <HStack w="full" justify="space-between">
                <Text fontWeight="medium" fontSize="md">
                  {jobStatus?.message || 'Exporting...'}
                </Text>
                <Text fontSize="sm" color="gray.600">
                  {(progress * 100).toFixed(0)}%
                </Text>
              </HStack>
              <Progress
                value={progress * 100}
                w="full"
                colorScheme="brand"
                size="sm"
                borderRadius="full"
              />
            </VStack>
          ) : (
            <HStack justify="space-between">
              <Text fontSize="sm" color="gray.600">
                Ready to export corrected images
              </Text>
              <Button
                colorScheme="brand"
                size="md"
                onClick={handleExportClick}
                isLoading={exportMutation.isLoading}
              >
                Export All
              </Button>
            </HStack>
          )}
        </Container>
      </Box>
      
      {/* Confirmation dialog */}
      <AlertDialog
        isOpen={isConfirmOpen}
        leastDestructiveRef={cancelRef}
        onClose={onConfirmClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Confirm Export
            </AlertDialogHeader>
            <AlertDialogBody>
              Are you sure you want to export all corrected images? This will process and save all images with the current corrections.
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onConfirmClose}>
                Cancel
              </Button>
              <Button colorScheme="brand" onClick={handleConfirmExport} ml={3}>
                Export
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
      
      {/* Export summary modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Export Summary</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            {exportSummary && (
              <VStack align="stretch" spacing={4}>
                <HStack justify="space-between">
                  <Text>Total Images:</Text>
                  <Text fontWeight="bold">{exportSummary.total_images}</Text>
                </HStack>
                <HStack justify="space-between">
                  <Text>Files Written:</Text>
                  <Text fontWeight="bold">{exportSummary.total_files_written}</Text>
                </HStack>
                <HStack justify="space-between">
                  <Text>Overwritten:</Text>
                  <Text fontWeight="bold">{exportSummary.overwritten_count}</Text>
                </HStack>
                <HStack justify="space-between">
                  <Text>New Files:</Text>
                  <Text fontWeight="bold">{exportSummary.new_files_count}</Text>
                </HStack>
                {exportSummary.failed_count > 0 && (
                  <HStack justify="space-between">
                    <Text color="red.500">Failed:</Text>
                    <Text fontWeight="bold" color="red.500">
                      {exportSummary.failed_count}
                    </Text>
                  </HStack>
                )}
                
                {exportSummary.errors.length > 0 && (
                  <Box>
                    <Text fontWeight="bold" mb={2}>Errors:</Text>
                    <List spacing={1} maxH="200px" overflowY="auto">
                      {exportSummary.errors.map((error, idx) => (
                        <ListItem key={idx} fontSize="sm" color="red.600">
                          • {error}
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </VStack>
            )}
          </ModalBody>
          <ModalFooter>
            <Button onClick={onClose}>Close</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  )
}

export default ExportBar

