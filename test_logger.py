"""
test_logger.py — quick sanity check for golfbot_logger.
Run with: python test_logger.py
No camera, robot, or other dependencies needed.
"""

from golfbot_logger import setup_logging, get_logger

setup_logging(level="DEBUG")
log = get_logger(__name__)

log.debug("DEBUG — fine detail, pipeline values, cm coords")
log.info("INFO — normal milestone, state transition")
log.warning("WARNING — recoverable issue, pose lost")
log.error("ERROR — TCP failure, motor fault")
log.critical("CRITICAL — unrecoverable, no camera found")