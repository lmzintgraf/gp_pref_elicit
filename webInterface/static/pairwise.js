if (winner != -1){
    var items = document.getElementsByClassName('single-item');
    items[winner-1].classList.add('moving-copy');
    setTimeout(function(){ items[winner-1].classList.remove('moving-copy'); }, 400);
}

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


function mouseEnterJob(event) {
    target_el(event).classList.add('pairwise-hovered-over');
}


function mouseLeaveJob(event) {
    target_el(event).classList.remove('pairwise-hovered-over');
    target_el(event).classList.remove('moving-copy');
}


function mouseClickJob(event) {

    // get the thing that was clicked
    var target = target_el(event);
    if (target.classList.contains('single-item-text')) {
        target = target.parentNode;
    }

    // get the item that was not clicked and make it invisible for some time
    if (target.getAttribute('ID') == 'left') {
        var not_clicked = 'right';
    } else {
        var not_clicked = 'left';
    }
    document.getElementById(not_clicked).classList.add('invisible');

    setTimeout(function(){ target.parentNode.submit(); }, 400);
}