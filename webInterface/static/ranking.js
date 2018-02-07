// this variable tells us which sorting area we're currently dragging over
// this is either unranked-items or ranked-items
var currDragOverArea = null;


function target_el(event) {
    // depends on the browser what we have to call
    var target_el = null;
    if (typeof event.srcElement !== 'undefined') {
        target_el = event.srcElement;
    }
    if (typeof event.target !== 'undefined') {
        target_el = event.target;
    }
    return target_el;
}

function get_style(target) {
    var style = null;
    if (typeof target.currentStyle != 'undefined') {
        style = target.currentStyle;
    }
    if (typeof window.getComputedStyle(target) != 'undefined') {
        style = window.getComputedStyle(target);
    }
    return style
}


// called when we start dragging an element
function dragStart(event){

    // allow the element to completely move somewhere else (not get copied)
    event.dataTransfer.effectAllowed = 'move';

    // the data we transfer is the ID of the element; this way we can drop it later
    event.dataTransfer.setData("Text", event.target.getAttribute('id'));

    // this is the item we drag
    var target = target_el(event);

    // create copy for ghost image
    var crt = target_el(event).cloneNode(true);

    // get position of target; then make it invisible; then move it manually
    target.style.zIndex = 1; // we have to do this first! otherwise it immediately drops
    marginTop = get_style(target).marginTop;
    marginLeft = get_style(target).marginLeft;
    targetWidth = target.getBoundingClientRect().width;
    targetHeight = target.getBoundingClientRect().height;
    target.classList.add('is-moving');
    target.style.marginTop = marginTop;
    target.style.marginLeft =  marginLeft;

    // change ID to prevent duplicates
    crt.setAttribute('ID', 'hidden-object');
    crt.classList.add('moving-copy');

    // put the ghost image somewhere so we can show it
    crt.style.position="absolute";
    document.getElementById('hidingSpot').appendChild(crt);
    event.dataTransfer.setDragImage(crt, targetWidth/2, targetHeight/2);

    dragOverItem(event);

    // (not sure why we have to return true)
    return true;
}

// https://stackoverflow.com/questions/5598743/finding-elements-position-relative-to-the-document
function getOffsetLeft(elem) {
    var offsetLeft = 0;
    do {
      if ( !isNaN( elem.offsetLeft ) )
      {
          offsetLeft += elem.offsetLeft;
      }
    } while( elem = elem.offsetParent );
    return offsetLeft;
}

// https://stackoverflow.com/questions/5598743/finding-elements-position-relative-to-the-document
function getOffsetTop(elem) {
    var offsetTop = 0;
    do {
      if ( !isNaN( elem.offsetTop ) )
      {
          offsetTop += elem.offsetTop;
      }
    } while( elem = elem.offsetParent );
    return offsetTop;
}


// called when we stop dragging an element
function dragEnd(event) {
    // reset style of all items that moved for our dragged object
    clearDragoverItems();
    // remove class of this element so we can change the style back
    target_el(event).classList.remove('is-moving');
    target_el(event).style.marginTop = "auto";

    // remove the hidden object
    var hidden_object = document.getElementById("hidden-object");
    hidden_object.parentNode.removeChild(hidden_object);
}


// called when we drag an element into another one
function dragEnter(event){

    // first check if we're still in the same sorting area (happens if we hovered over an item in that same area)
    var stayed = target_el(event).getAttribute('ID') == currDragOverArea;
    if (!stayed) {
        // find out if we entered a sorting area
        if (target_el(event).getAttribute('ID') == 'ranked-items') {
            // update the class we're currently in
            currDragOverArea = 'ranked-items';
            // re-set the style of the items
            clearDragoverItems();
        } else if (target_el(event).getAttribute('ID') == 'unranked-items') {
            // update the class we're currently in
            currDragOverArea = 'unranked-items';
            // re-set the style of the items
            clearDragoverItems();
        }
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';

    return true;
}


// called when we drag over a drop area
function dragOver(event) {

    event.preventDefault();

    // check if we're hovering over an item
    if (target_el(event).classList.contains('single-item')) {
        dragOverItem(event);
    }
}


// called when we drag an element over another draggable element
function dragOverItem(event) {

    // make all other items normal again!
    clearDragoverItems();

    // get the target element (over which we drag)
    var target = target_el(event);

    // get the container of the jobs
    var parent = target.parentNode;
    while (parent != null && parent.getAttribute('ID') != 'ranked-items' && parent.getAttribute('ID') != 'unranked-items') {
        parent = parent.parentNode;
    }

    // get the children, i.e., jobs
    var children = parent.children;

    // y position of draggable element
    var drag_top = event.clientY;

    // loop through children until we find the one we're hovering over
    var moved_child = false;
    var move_next = false;
    for (var i=0; i<children.length; i++) {
        var child = children[i];

        // height of child
        var child_height = child.getBoundingClientRect().height;
        // top position of child
        var child_top = child.getBoundingClientRect().top;
        // bottom position of child
        var child_bottom = child.getBoundingClientRect().bottom;
        // see whether we are hovering over the child
        if (!child.classList.contains('is-moving') && !moved_child) {
            if (child_top < drag_top && drag_top < child_bottom) {
                child.classList.add('dragover-item');
                moved_child = true;
            }
        }
    }

}

// called when we drop an element
function handleDrop(event){

    // get the ID of the element we transferred
    var src = event.dataTransfer.getData("Text");
    // get the element we transfer
    var insertedItem = document.getElementById(src);

    // where we drop it into
    var target = target_el(event);
    while (target.getAttribute('ID') != 'ranked-items' && target.getAttribute('ID') != 'unranked-items') {
        target = target.parentElement;
    }

    // now we need to figure out where to drop it...
    // (1) it's dragged on top of something
    var dragovers = document.getElementsByClassName('dragover-item');
    if (dragovers.length > 0) {
        dragovers[0].insertAdjacentElement('beforeBegin', insertedItem);
    } else {
        // (2) it's dragged on the bottom of something
        dragovers = document.getElementsByClassName('dragover-item');
        if (dragovers.length > 0) {
            dragovers[0].insertAdjacentElement('afterEnd', insertedItem);
        } else {
            // (3) none of the above; then just put it at the bottom
            target.appendChild(insertedItem);
        }
    }

    // this is only for jobs: we want to change their appearance depending on where we put them
    if (insertedItem.classList.contains('job')) {
        if (target.getAttribute('ID') == 'ranked-items') {
            minimise_job(insertedItem);
        } else {
            maximise_job(insertedItem);
        }
    }

    // reset style of all items that moved for our dragged object
    clearDragoverItems();

    // make the dropped item appear differently for a short amount of time
    insertedItem.classList.add('just-dropped');
    insertedItem.style.zIndex = '1';
    insertedItem.style.marginLeft = 'auto';
    setTimeout(function(){
        insertedItem.style.zIndex = '0';
        insertedItem.classList.remove('just-dropped');
        }, 400);

    // ?
    event.stopPropagation();

    // (note sure why we have to return false here)
    return false;
}


// make the description of a job tiny
function minimise_job(job) {
    job.classList.add('sorted-in-job');
}

// make the description of a job large again
function maximise_job(job) {
    job.classList.remove('sorted-in-job');
}


// drag gp_utilities: clear drag-over-items style
function clearDragoverItems() {
    var items = document.getElementsByClassName('single-item');
    for (var i=0; i < items.length; i++) {
        items[i].classList.remove('dragover-item');
    }
}


// called when the user submits the ranking
function handleSubmit(button_type) {

    // check if there are no more unranked items
    unranked_items = document.getElementById('unranked-items').children;
    if (unranked_items.length > 0) {
        window.alert('Please sort in all items first.');
    }
    else {
         //get the ranking and save it
        var item_ids = [];
        var ranked_items = document.getElementById("ranked-items").children;
        for (var j=0, item; item = ranked_items[j]; j++) {
            item_ids.push(ranked_items[j].getAttribute('id'));
        }
        document.getElementsByName("rankingResult")[0].value = item_ids;

        var btn = document.getElementsByName('buttonType')[0];
        if (btn) {
            btn.value = button_type;
        }

        // now submit the form
        document.getElementById("subForm").submit();
    }

}
